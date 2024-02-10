// script.js
$(document).ready(function() {
    var source = new EventSource("/video_feed");

    source.onmessage = function(event) {
        var frames = event.data.split("--frame");
        var detectedFrame = frames[1].split("\r\n\r\n")[1];
        var blurredFrame = frames[3].split("\r\n\r\n")[1];

        document.getElementById("detected_frame").src = "data:image/jpeg;base64," + detectedFrame;
        document.getElementById("blurred_frame").src = "data:image/jpeg;base64," + blurredFrame;
    };
});