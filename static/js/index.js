document.addEventListener('DOMContentLoaded', function () {
    var synth = window.speechSynthesis;
    var msg = new SpeechSynthesisUtterance();
    var voices = synth.getVoices();
    msg.voice = voices[0];
    msg.rate = 1;
    msg.pitch = 1;

    document.getElementById('chatbot-form-btn').addEventListener('click', function (e) {
        e.preventDefault();
        showLoader(); // Show loader when the form is submitted
        document.getElementById('chatbot-form').submit();
    });

    document.getElementById('chatbot-form-btn-clear').addEventListener('click', function (e) {
        e.preventDefault();
        document.getElementById('chatPanel').querySelector('.media-list').innerHTML = '';
    });

    document.getElementById('chatbot-form').submit = function (e) {
        // e.preventDefault();

        var message = document.getElementById('messageText').value;
        var mediaList = document.querySelector('.media-list');
        var chatPanel = document.getElementById('chatPanel');

        mediaList.innerHTML += '<li class="media"><div class="media-body"><div class="media"><div style="text-align:right; color: #2EFE2E" class="media-body">' +
            message + '<hr/></div></div></div></li>';

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/ask', true);
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');

        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4 && xhr.status === 200) {
                document.getElementById('messageText').value = '';
                var response = JSON.parse(xhr.responseText);
                var answer = response.answer;

                mediaList.innerHTML += '<li class="media"><div class="media-body"><div class="media"><div style="color: white" class="media-body">' +
                    answer + '<hr/></div></div></div></li>';

                document.querySelector('.fixed-panel').scroll({
                    top: document.querySelector('.fixed-panel').scrollHeight,
                    behavior: 'smooth'
                });

                msg.text = answer;
                speechSynthesis.speak(msg);

                hideLoader(); // Hide loader when the response is received
            }
        };

        xhr.onerror = function (error) {
            console.log(error);
            hideLoader(); // Hide loader in case of an error
        };

        xhr.send(new URLSearchParams(new FormData(this)));
    };

    function showLoader() {
        document.getElementById('loader').style.display = 'block';
    }

    function hideLoader() {
        document.getElementById('loader').style.display = 'none';
    }
});
