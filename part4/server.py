"""
server.py
CherryPy web application: upload a raw email CSV, clean it, engineer features,
train/evaluate a TF-IDF + Naive Bayes spam classifier, and render the results
via Genshi templates.

Run:
    python3 server.py
Then open http://localhost:8080/
"""
import io
import os

import cherrypy
from genshi.template import TemplateLoader

import processing as proc
from logger_setup import get_logger

logger = get_logger()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(CUR_DIR, "templates")
STATIC_DIR = os.path.join(CUR_DIR, "static")

loader = TemplateLoader(TEMPLATES_DIR, auto_reload=True)


def render(template_name, **kwargs):
    tmpl = loader.load(template_name)
    stream = tmpl.generate(**kwargs)
    return stream.render("html", doctype="html")


class SpamDetectionApp:

    def __init__(self, model_path, api_url, api_key):
        self.model_path = model_path
        self.api_url = api_url
        self.api_key = api_key

    @cherrypy.expose
    def index(self, error=None):
        client_ip = cherrypy.request.remote.ip
        logger.info("ACTION POST /part4 from %s: model_path=%s llm_api_url=%s key_submitted=%s", client_ip, self.model_path, self.api_url or "(default)", bool(self.api_key))
        if not self.model_path:
            logger.warning("POST /part4 from %s: no model_path provided", client_ip)
            return render("index.html", error="No Part 3 model to explain. Run Part 3 first.")
        try:
            result = proc.run_pipeline(
                self.model_path,
                api_url=self.api_url,
                api_key=self.api_key,
            )
        except FileNotFoundError as exc:
            logger.warning("POST from %s: %s", client_ip, exc)
            return render("index.html", error=str(exc))
        except Exception as exc:
            logger.exception("POST from %s: unexpected error processing %s", client_ip, self.model_path)
            return render("index.html", error=f"Unexpected error: {exc}")
        logger.info("POST from %s: success, using_mock=%s", client_ip, result["using_mock"])
        return render("results.html", **result)


def main():

    # get model path, api url and key
    from configobj import ConfigObj
    file_path = os.path.join("meta.ini")
    if not os.path.exists(file_path):
        logger.exception("Missing configuration.")
        return False
    # read mailsource configuration
    system_conf_obj = ConfigObj(file_path)
    API_URL = system_conf_obj["llm"]["api_url"] 
    API_KEY = system_conf_obj["llm"]["api_key"] 
    MODEL_PATH = system_conf_obj["llm"]["model_path"] 
    cherrypy.config.update({
        "server.socket_host": "0.0.0.0",
        "server.socket_port": int(os.environ.get("PORT", 8080)),
        "server.max_request_body_size": 300 * 1024 * 1024,  # 300 MB, for large CSVs
        "server.thread_pool": 8,
        "log.screen": True,
    })
    conf = {
        "/": {
            "tools.sessions.on": False,
            "request.show_tracebacks": False,
        },
        "/static": {
            "tools.staticdir.on": True,
            "tools.staticdir.dir": STATIC_DIR,
        },
    }
    logger.info("ACTION server startup: listening on %s:%s", cherrypy.config["server.socket_host"], cherrypy.config["server.socket_port"])
    try:
        cherrypy.quickstart(SpamDetectionApp(MODEL_PATH, API_URL, API_KEY), "/", conf)
    finally:
        logger.info("ACTION server shutdown")


if __name__ == "__main__":
    main()
