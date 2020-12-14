from base64 import b64decode

from IPython.display import clear_output, display, Javascript

from . import timestampstr, TimeIt


def record_video(filename=f'video_{timestampstr()}.mp4'):
    # https://stackoverflow.com/questions/62529304/is-there-any-way-to-capture-live-video-using-webcam-in-google-colab
    # This function uses the take_photo() function provided by the Colab team as a
    # starting point, along with a bunch of stuff from Stack overflow, and some sample code
    # from: https://developer.mozilla.org/enUS/docs/Web/API/MediaStream_Recording_API

    from google.colab.output import eval_js

    js = Javascript("""
    async function recordVideo() {
      const options = { mimeType: 'video/webm; codecs=vp9' };
      const div = document.createElement('div');
      const capture = document.createElement('button');
      const stopCapture = document.createElement('button');
      capture.textContent = 'Start Recording';
      capture.style.background = 'green';
      capture.style.color = 'white';

      stopCapture.textContent = 'Stop Recording';
      stopCapture.style.background = 'red';
      stopCapture.style.color = 'white';
      div.appendChild(capture);

      const video = document.createElement('video');
      const recordingVid = document.createElement('video');
      video.style.display = 'block';

      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      let recorder = new MediaRecorder(stream, options);
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      await new Promise((resolve) => {
        capture.onclick = resolve;
      });
      recorder.start();
      capture.replaceWith(stopCapture);
      await new Promise((resolve) => stopCapture.onclick = resolve);
      recorder.stop();

      let recData = await new Promise((resolve) => recorder.ondataavailable = resolve);
      let arrBuff = await recData.data.arrayBuffer();
      stream.getVideoTracks()[0].stop();
      div.remove();

      let binaryString = '';
      let bytes = new Uint8Array(arrBuff);
      bytes.forEach((byte) => {
        binaryString += String.fromCharCode(byte);
      })
      return btoa(binaryString);
      // let bytes = new Uint8Array(arrBuff);
      // return btoa(bytes
      //     .map((byte) => String.fromCharCode(byte))
      //     .join(''));
    }
    """)
    display(js)
    data = eval_js('recordVideo({})')
    print('Finished recording video.')
    binary = b64decode(data)
    with TimeIt():
        with open(filename, 'wb') as f:
            print(f'Saving video binary at {filename}')
            f.write(binary)
    return filename


def select_index_prompt(choices, description=None):
    valid_choices = range(len(choices))

    def is_valid_choice(choice):
        return choice is not None and choice.isdigit() and int(choice) in valid_choices

    selected_index = None
    while not is_valid_choice(selected_index):
        clear_output()

        # https://stackoverflow.com/questions/51642478/inconsistent-output-in-google-colab
        output = []

        if selected_index is not None:
            output.append(f'Selected index is not valid: \'{selected_index}\'')

        if description:
            output.append(description)
        output.append('Please, select one of the following indices:')
        for idx, display_name in enumerate(choices):
            output.append(f' [{idx}] - {display_name}')
        output.append('Selected index: ')

        selected_index = input('\n'.join(output))

    return int(selected_index)
