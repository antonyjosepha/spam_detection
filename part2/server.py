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

    @cherrypy.expose
    def index(self, error=None):
        logger.info("ACTION GET / from %s%s", cherrypy.request.remote.ip,
                    f" (error shown: {error})" if error else "")
        return render("index.html", error=error)

    @cherrypy.expose
    def upload(self, csvfile=None):
        client_ip = cherrypy.request.remote.ip

        if csvfile is None or not getattr(csvfile, "filename", ""):
            logger.warning("ACTION POST /upload from %s: no file provided", client_ip)
            return render("index.html", error="Please choose a CSV file to upload.")

        logger.info("ACTION POST /upload from %s: filename=%s", client_ip, csvfile.filename)

        if not csvfile.filename.lower().endswith(".csv"):
            logger.warning("POST /upload from %s: rejected non-csv filename=%s",
                            client_ip, csvfile.filename)
            return render("index.html", error="Please upload a file with a .csv extension.")

        try:
            raw_bytes = csvfile.file.read()
            logger.info("POST /upload from %s: read %d bytes for %s",
                        client_ip, len(raw_bytes), csvfile.filename)
            file_like = io.BytesIO(raw_bytes)
            result = proc.run_pipeline(file_like)
        except proc.PipelineError as exc:
            logger.warning("POST /upload from %s: pipeline rejected file %s (%s)",
                            client_ip, csvfile.filename, exc)
            return render("index.html", error=str(exc))
        except Exception as exc:
            logger.exception("POST /upload from %s: unexpected error processing %s",
                              client_ip, csvfile.filename)
            return render(
                "index.html",
                error=f"Unexpected error while processing the file: {exc}",
            )

        logger.info("POST /part2 from %s: success, AUC=%.4f R2(reg)=%.4f",
                    client_ip, result["classification"]["auc"], result["regression"]["r2_lr"])
        return render(
            "results.html",
            cleaned_csv_path=result["cleaned_path"],
            run_dir=result["run_dir"],
            feature_names=result["feature_names"],
            regression=result["regression"],
            classification=result["classification"],
            regularization=result["regularization"],
            bootstrap=result["bootstrap"],
        )


def main():
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

    logger.info("ACTION server startup: listening on %s:%s",
                cherrypy.config["server.socket_host"], cherrypy.config["server.socket_port"])
    try:
        cherrypy.quickstart(SpamDetectionApp(), "/", conf)
    finally:
        logger.info("ACTION server shutdown")


if __name__ == "__main__":
    main()
