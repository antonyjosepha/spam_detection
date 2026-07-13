$(function () {
    // ---- Results page: tab switching ----
    $(".tab-btn").on("click", function () {
        var target = $(this).data("tab");
        $(".tab-btn").removeClass("active");
        $(this).addClass("active");
        $(".tab-panel").removeClass("active");
        $("#" + target).addClass("active");
    });

    // ---- Upload page: show spinner + disable button on submit ----
    $("#uploadForm").on("submit", function () {
        var fileInput = $("#csvfile")[0];
        if (!fileInput || !fileInput.files.length) {
            return false;
        }
        $("#submitBtn").prop("disabled", true).text("Processing...");
        $("#loading").removeClass("hidden");
    });

    // ---- Results page: predictions table search filter ----
    $("#predSearch").on("keyup", function () {
        var query = $(this).val().toLowerCase();
        $("#predTable tbody tr").each(function () {
            var subject = $(this).find("td").first().text().toLowerCase();
            $(this).toggle(subject.indexOf(query) > -1);
        });
    });
});
