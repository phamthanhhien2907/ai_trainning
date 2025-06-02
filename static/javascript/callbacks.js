(function () {
  // Audio context initialization
  let mediaRecorder = undefined;
  let audioChunks, audioBlob, stream, audioRecorded;
  const ctx = new AudioContext();
  let currentAudioForPlaying;
  let lettersOfWordAreCorrect = [];

  // UI-related variables
  const page_title = "AI Pronunciation Trainer";
  const accuracy_colors = ["green", "orange", "red"];
  let badScoreThreshold = 30;
  let mediumScoreThreshold = 70;
  let currentSample = 0;
  let currentScore = 0;
  let sample_difficult = 0;
  let scoreMultiplier = 1;
  let playAnswerSounds = true;
  let isNativeSelectedForPlayback = true;
  let isRecording = false;
  let serverIsInitialized = false;
  let serverWorking = true;
  let languageFound = true;
  let currentSoundRecorded = false;
  let currentText, currentIpa, real_transcripts_ipa, matched_transcripts_ipa;
  let wordCategories;
  let startTime, endTime;

  // API related variables
  let AILanguage = "de";
  let STScoreAPIKey = "rll5QsTiv83nti99BW6uCmvs9BDVxSB39SVFceYb";
  let apiMainPathSample = "http://127.0.0.1:3000";
  let apiMainPathSTS = "http://127.0.0.1:3000";

  // Variables to playback accuracy sounds
  let soundsPath = "/static";
  let soundFileGood = null;
  let soundFileOkay = null;
  let soundFileBad = null;

  // Speech generation
  var synth = window.speechSynthesis;
  let voice_idx = 0;
  let voice_synth = null;

  // UI control functions
  const unblockUI = () => {
    const recordAudio = document.getElementById("recordAudio");
    const playSampleAudio = document.getElementById("playSampleAudio");
    const buttonNext = document.getElementById("buttonNext");
    const nextButtonDiv = document.getElementById("nextButtonDiv");
    const originalScript = document.getElementById("original_script");
    const playRecordedAudio = document.getElementById("playRecordedAudio");

    if (recordAudio) recordAudio.classList.remove("disabled");
    if (playSampleAudio) playSampleAudio.classList.remove("disabled");
    if (buttonNext) buttonNext.style["background-color"] = "#58636d";
    if (nextButtonDiv) nextButtonDiv.classList.remove("disabled");
    if (originalScript) originalScript.classList.remove("disabled");
    if (currentSoundRecorded && playRecordedAudio)
      playRecordedAudio.classList.remove("disabled");
    console.log(
      "unblockUI: recordAudio disabled:",
      recordAudio?.classList.contains("disabled")
    );
  };

  const blockUI = () => {
    const recordAudio = document.getElementById("recordAudio");
    const playSampleAudio = document.getElementById("playSampleAudio");
    const buttonNext = document.getElementById("buttonNext");
    const originalScript = document.getElementById("original_script");
    const playRecordedAudio = document.getElementById("playRecordedAudio");

    if (recordAudio) recordAudio.classList.add("disabled");
    if (playSampleAudio) playSampleAudio.classList.add("disabled");
    if (buttonNext) buttonNext.style["background-color"] = "#adadad";
    if (originalScript) originalScript.classList.add("disabled");
    if (playRecordedAudio) playRecordedAudio.classList.add("disabled");
    console.log(
      "blockUI: recordAudio disabled:",
      recordAudio?.classList.contains("disabled")
    );
  };

  const UIError = (errorMessage = null) => {
    blockUI();
    const buttonNext = document.getElementById("buttonNext");
    const recordedIpaScript = document.getElementById("recorded_ipa_script");
    const singleWordIpa = document.getElementById("single_word_ipa");
    const ipaScript = document.getElementById("ipa_script");
    const mainTitle = document.getElementById("main_title");
    const originalScript = document.getElementById("original_script");

    if (buttonNext) buttonNext.style["background-color"] = "#58636d";
    if (recordedIpaScript) recordedIpaScript.innerHTML = "";
    if (singleWordIpa) singleWordIpa.innerHTML = "Error";
    if (ipaScript) ipaScript.innerHTML = "Error";
    if (mainTitle) mainTitle.innerHTML = "Server Error";
    if (originalScript)
      originalScript.innerHTML =
        errorMessage ||
        "Server error. Please try again or check server logs for details.";
    setTimeout(unblockUI, 2000);
    console.log(
      "UIError: recordAudio disabled:",
      document.getElementById("recordAudio")?.classList.contains("disabled")
    );
  };

  const UINotSupported = () => {
    unblockUI();
    const mainTitle = document.getElementById("main_title");
    if (mainTitle) mainTitle.innerHTML = "Browser unsupported";
  };

  const UIRecordingError = () => {
    unblockUI();
    const mainTitle = document.getElementById("main_title");
    if (mainTitle)
      mainTitle.innerHTML =
        "Recording error, please try again or restart page.";
    startMediaDevice();
  };

  // Application state functions
  function updateScore(currentPronunciationScore) {
    if (isNaN(currentPronunciationScore)) return;
    currentScore += currentPronunciationScore * scoreMultiplier;
    currentScore = Math.round(currentScore);
  }

  const cacheSoundFiles = async () => {
    await fetch(soundsPath + "/ASR_good.wav")
      .then((data) => data.arrayBuffer())
      .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
      .then((decodeAudioData) => {
        soundFileGood = decodeAudioData;
      });
    await fetch(soundsPath + "/ASR_okay.wav")
      .then((data) => data.arrayBuffer())
      .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
      .then((decodeAudioData) => {
        soundFileOkay = decodeAudioData;
      });
    await fetch(soundsPath + "/ASR_bad.wav")
      .then((data) => data.arrayBuffer())
      .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
      .then((decodeAudioData) => {
        soundFileBad = decodeAudioData;
      });
  };

  const getNextSample = async () => {
    blockUI();
    try {
      if (!serverIsInitialized) await initializeServer();
      if (!serverWorking) {
        UIError();
        return;
      }
      if (soundFileBad == null) await cacheSoundFiles();

      updateScore(
        parseFloat(
          document.getElementById("pronunciation_accuracy")?.innerHTML || "0"
        )
      );

      const mainTitle = document.getElementById("main_title");
      if (mainTitle) mainTitle.innerHTML = "Processing new sample...";

      if (document.getElementById("lengthCat1")?.checked) {
        sample_difficult = 0;
        scoreMultiplier = 1.3;
      } else if (document.getElementById("lengthCat2")?.checked) {
        sample_difficult = 1;
        scoreMultiplier = 1;
      } else if (document.getElementById("lengthCat3")?.checked) {
        sample_difficult = 2;
        scoreMultiplier = 1.3;
      } else if (document.getElementById("lengthCat4")?.checked) {
        sample_difficult = 3;
        scoreMultiplier = 1.6;
      }

      const response = await fetch(apiMainPathSample + "/getSample", {
        method: "POST",
        body: JSON.stringify({
          category: sample_difficult.toString(),
          language: AILanguage,
        }),
        headers: {
          "X-Api-Key": STScoreAPIKey,
          "Content-Type": "application/json",
        },
      });
      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      const originalScript = document.getElementById("original_script");
      currentText = data.real_transcript;
      if (originalScript)
        originalScript.innerHTML = currentText || "No transcript available";

      currentIpa = data.ipa_transcript;
      const ipaScript = document.getElementById("ipa_script");
      if (ipaScript)
        ipaScript.innerHTML = currentIpa
          ? "/ " + currentIpa + " /"
          : "No IPA available";

      const recordedIpaScript = document.getElementById("recorded_ipa_script");
      const pronunciationAccuracy = document.getElementById(
        "pronunciation_accuracy"
      );
      const singleWordIpa = document.getElementById("single_word_ipa");
      const sectionAccuracy = document.getElementById("section_accuracy");
      const translatedScript = document.getElementById("translated_script");
      const playRecordedAudio = document.getElementById("playRecordedAudio");

      if (recordedIpaScript) recordedIpaScript.innerHTML = "";
      if (pronunciationAccuracy) pronunciationAccuracy.innerHTML = "";
      if (singleWordIpa) singleWordIpa.innerHTML = "Reference | Spoken";
      if (sectionAccuracy)
        sectionAccuracy.innerHTML =
          "| Score: " +
          currentScore.toString() +
          " - (" +
          currentSample.toString() +
          ")";
      currentSample += 1;

      if (mainTitle) mainTitle.innerHTML = page_title;
      if (translatedScript)
        translatedScript.innerHTML =
          data.transcript_translation ||
          data.transcript_translated ||
          "No translation available";
      currentSoundRecorded = false;
      unblockUI();
      if (playRecordedAudio) playRecordedAudio.classList.add("disabled");
    } catch (e) {
      console.error("Fetch sample error:", e);
      UIError(`Failed to fetch sample: ${e.message}`);
      unblockUI();
    }
  };

  const updateRecordingState = async () => {
    if (isRecording) await stopRecording();
    else await recordSample();
  };

  const generateWordModal = (word_idx) => {
    const singleWordIpa = document.getElementById("single_word_ipa");
    if (singleWordIpa) {
      singleWordIpa.innerHTML =
        wrapWordForPlayingLink(
          real_transcripts_ipa[word_idx],
          word_idx,
          false,
          "black"
        ) +
        " | " +
        wrapWordForPlayingLink(
          matched_transcripts_ipa[word_idx],
          word_idx,
          true,
          accuracy_colors[parseInt(wordCategories[word_idx])]
        );
    }
  };

  const recordSample = async () => {
    const mainTitle = document.getElementById("main_title");
    const recordIcon = document.getElementById("recordIcon");
    if (mainTitle)
      mainTitle.innerHTML = "Recording... click again when done speaking";
    if (recordIcon) recordIcon.innerHTML = "pause_presentation";
    blockUI();
    const recordAudio = document.getElementById("recordAudio");
    if (recordAudio) recordAudio.classList.remove("disabled");
    audioChunks = [];
    isRecording = true;
    if (!mediaRecorder) {
      try {
        const _stream = await navigator.mediaDevices.getUserMedia(
          mediaStreamConstraints
        );
        stream = _stream;
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };
        mediaRecorder.onstop = async () => {
          const recordIcon = document.getElementById("recordIcon");
          if (recordIcon) recordIcon.innerHTML = "mic";
          blockUI();
          audioBlob = new Blob(audioChunks, { type: "audio/ogg;" });
          let audioUrl = URL.createObjectURL(audioBlob);
          audioRecorded = new Audio(audioUrl);
          let audioBase64 = await convertBlobToBase64(audioBlob);
          let minimumAllowedLength = 6;
          if (audioBase64.length < minimumAllowedLength) {
            setTimeout(UIRecordingError, 50);
            return;
          }
          try {
            let text =
              document.getElementById("original_script")?.innerHTML || "";
            if (typeof text !== "string") {
              console.error("text is not a string:", text);
              UIError("Invalid text format for pronunciation analysis");
              return;
            }
            text = text
              .replace(/<[^>]*>?/gm, "")
              .trim()
              .replace(/\s\s+/g, " ");
            if (!text) {
              UIError("No text provided for pronunciation analysis");
              return;
            }
            currentText = [text];
            const response = await fetch(
              apiMainPathSTS + "/GetAccuracyFromRecordedAudio",
              {
                method: "POST",
                body: JSON.stringify({
                  title: currentText[0],
                  base64Audio: audioBase64,
                  language: AILanguage,
                }),
                headers: {
                  "X-Api-Key": STScoreAPIKey,
                  "Content-Type": "application/json",
                },
              }
            );
            if (!response.ok)
              throw new Error(`HTTP error! Status: ${response.status}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            if (playAnswerSounds)
              playSoundForAnswerAccuracy(
                parseFloat(data.pronunciation_accuracy)
              );
            const recordedIpaScript = document.getElementById(
              "recorded_ipa_script"
            );
            const recordAudio = document.getElementById("recordAudio");
            const mainTitle = document.getElementById("main_title");
            const pronunciationAccuracy = document.getElementById(
              "pronunciation_accuracy"
            );
            const ipaScript = document.getElementById("ipa_script");
            const originalScript = document.getElementById("original_script");
            const playRecordedAudio =
              document.getElementById("playRecordedAudio");

            if (recordedIpaScript)
              recordedIpaScript.innerHTML = "/ " + data.ipa_transcript + " /";
            if (recordAudio) recordAudio.classList.add("disabled");
            if (mainTitle) mainTitle.innerHTML = page_title;
            if (pronunciationAccuracy)
              pronunciationAccuracy.innerHTML =
                data.pronunciation_accuracy + "%";
            if (ipaScript) ipaScript.innerHTML = data.real_transcripts_ipa;
            lettersOfWordAreCorrect =
              data.is_letter_correct_all_words.split(" ");
            startTime = data.start_time;
            endTime = data.end_time;
            real_transcripts_ipa = data.real_transcripts_ipa.split(" ");
            matched_transcripts_ipa = data.matched_transcripts_ipa.split(" ");
            wordCategories = data.pair_accuracy_category.split(" ");
            let currentTextWords = currentText[0].split(" ");
            let coloredWords = "";
            for (
              let word_idx = 0;
              word_idx < currentTextWords.length;
              word_idx++
            ) {
              let wordTemp = "";
              for (
                let letter_idx = 0;
                letter_idx < currentTextWords[word_idx].length;
                letter_idx++
              ) {
                let letter_is_correct =
                  lettersOfWordAreCorrect[word_idx][letter_idx] === "1";
                let color_letter = letter_is_correct ? "green" : "red";
                wordTemp += `<font color="${color_letter}">${currentTextWords[word_idx][letter_idx]}</font>`;
              }
              coloredWords +=
                " " + wrapWordForIndividualPlayback(wordTemp, word_idx);
            }
            if (originalScript) originalScript.innerHTML = coloredWords;
            currentSoundRecorded = true;
            unblockUI();
            if (playRecordedAudio)
              playRecordedAudio.classList.remove("disabled");
          } catch (e) {
            console.error("Audio processing error:", e);
            UIError(`Failed to process audio: ${e.message}`);
          }
        };
      } catch (e) {
        console.error("Media device error on record:", e);
        if (
          e.name === "NotAllowedError" ||
          e.name === "PermissionDeniedError"
        ) {
          UIError(
            "Microphone access denied. Please allow microphone permissions in Chrome."
          );
        } else {
          UIError(`Failed to access media device: ${e.message}`);
        }
        return;
      }
    }
    try {
      mediaRecorder.start();
    } catch (error) {
      console.error("Error starting recorder:", error);
      UIRecordingError();
    }
  };

  const changeLanguage = (language, generateNewSample = false) => {
    const voices = synth.getVoices();
    AILanguage = language;
    languageFound = false;
    let languageIdentifier, languageName;
    const languageBox = document.getElementById("languageBox");
    switch (language) {
      case "de":
        if (languageBox) languageBox.innerHTML = "German";
        languageIdentifier = "de";
        languageName = "Anna";
        break;
      case "en":
        if (languageBox) languageBox.innerHTML = "English";
        languageIdentifier = "en";
        languageName = "Daniel";
        break;
    }
    for (let idx = 0; idx < voices.length; idx++) {
      if (
        voices[idx].lang.slice(0, 2) === languageIdentifier &&
        voices[idx].name === languageName
      ) {
        voice_synth = voices[idx];
        languageFound = true;
        break;
      }
    }
    if (!languageFound) {
      for (let idx = 0; idx < voices.length; idx++) {
        if (voices[idx].lang.slice(0, 2) === languageIdentifier) {
          voice_synth = voices[idx];
          languageFound = true;
          break;
        }
      }
    }
    if (generateNewSample) getNextSample();
  };

  // Speech-To-Score function
  const mediaStreamConstraints = {
    audio: {
      channelCount: 1,
      sampleRate: 48000,
    },
  };

  const startMediaDevice = async () => {
    let retries = 3;
    let success = false;
    while (retries > 0 && !success) {
      try {
        const _stream = await navigator.mediaDevices.getUserMedia(
          mediaStreamConstraints
        );
        stream = _stream;
        if (mediaRecorder === undefined) {
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
          };
          mediaRecorder.onstop = async () => {
            const recordIcon = document.getElementById("recordIcon");
            if (recordIcon) recordIcon.innerHTML = "mic";
            blockUI();
            audioBlob = new Blob(audioChunks, { type: "audio/ogg;" });
            let audioUrl = URL.createObjectURL(audioBlob);
            audioRecorded = new Audio(audioUrl);
            let audioBase64 = await convertBlobToBase64(audioBlob);
            let minimumAllowedLength = 6;
            if (audioBase64.length < minimumAllowedLength) {
              setTimeout(UIRecordingError, 50);
              return;
            }
            try {
              let text =
                document.getElementById("original_script")?.innerHTML || "";
              if (typeof text !== "string") {
                console.error("text is not a string:", text);
                UIError("Invalid text format for pronunciation analysis");
                return;
              }
              text = text
                .replace(/<[^>]*>?/gm, "")
                .trim()
                .replace(/\s\s+/g, " ");
              if (!text) {
                UIError("No text provided for pronunciation analysis");
                return;
              }
              currentText = [text];
              const response = await fetch(
                apiMainPathSTS + "/GetAccuracyFromRecordedAudio",
                {
                  method: "POST",
                  body: JSON.stringify({
                    title: currentText[0],
                    base64Audio: audioBase64,
                    language: AILanguage,
                  }),
                  headers: {
                    "X-Api-Key": STScoreAPIKey,
                    "Content-Type": "application/json",
                  },
                }
              );
              if (!response.ok)
                throw new Error(`HTTP error! Status: ${response.status}`);
              const data = await response.json();
              if (data.error) throw new Error(data.error);
              if (playAnswerSounds)
                playSoundForAnswerAccuracy(
                  parseFloat(data.pronunciation_accuracy)
                );
              const recordedIpaScript = document.getElementById(
                "recorded_ipa_script"
              );
              const recordAudio = document.getElementById("recordAudio");
              const mainTitle = document.getElementById("main_title");
              const pronunciationAccuracy = document.getElementById(
                "pronunciation_accuracy"
              );
              const ipaScript = document.getElementById("ipa_script");
              const originalScript = document.getElementById("original_script");
              const playRecordedAudio =
                document.getElementById("playRecordedAudio");

              if (recordedIpaScript)
                recordedIpaScript.innerHTML = "/ " + data.ipa_transcript + " /";
              if (recordAudio) recordAudio.classList.add("disabled");
              if (mainTitle) mainTitle.innerHTML = page_title;
              if (pronunciationAccuracy)
                pronunciationAccuracy.innerHTML =
                  data.pronunciation_accuracy + "%";
              if (ipaScript) ipaScript.innerHTML = data.real_transcripts_ipa;
              lettersOfWordAreCorrect =
                data.is_letter_correct_all_words.split(" ");
              startTime = data.start_time;
              endTime = data.end_time;
              real_transcripts_ipa = data.real_transcripts_ipa.split(" ");
              matched_transcripts_ipa = data.matched_transcripts_ipa.split(" ");
              wordCategories = data.pair_accuracy_category.split(" ");
              let currentTextWords = currentText[0].split(" ");
              let coloredWords = "";
              for (
                let word_idx = 0;
                word_idx < currentTextWords.length;
                word_idx++
              ) {
                let wordTemp = "";
                for (
                  let letter_idx = 0;
                  letter_idx < currentTextWords[word_idx].length;
                  letter_idx++
                ) {
                  let letter_is_correct =
                    lettersOfWordAreCorrect[word_idx][letter_idx] === "1";
                  let color_letter = letter_is_correct ? "green" : "red";
                  wordTemp += `<font color="${color_letter}">${currentTextWords[word_idx][letter_idx]}</font>`;
                }
                coloredWords +=
                  " " + wrapWordForIndividualPlayback(wordTemp, word_idx);
              }
              if (originalScript) originalScript.innerHTML = coloredWords;
              currentSoundRecorded = true;
              unblockUI();
              if (playRecordedAudio)
                playRecordedAudio.classList.remove("disabled");
            } catch (e) {
              console.error("Audio processing error:", e);
              UIError(`Failed to process audio: ${e.message}`);
            }
          };
        }
        success = true;
      } catch (e) {
        console.error("Media device error:", e);
        if (
          e.name === "NotAllowedError" ||
          e.name === "PermissionDeniedError"
        ) {
          UIError(
            "Microphone access denied. Please allow microphone permissions in Chrome."
          );
          break;
        } else {
          UIError(`Failed to access media device: ${e.message}`);
          retries--;
          if (retries > 0) {
            await new Promise((resolve) => setTimeout(resolve, 1000));
          }
        }
      }
    }
    if (!success) {
      UIError("Failed to initialize microphone after multiple attempts.");
    }
  };
  startMediaDevice();

  // Audio playback
  const playSoundForAnswerAccuracy = async (accuracy) => {
    currentAudioForPlaying = soundFileGood;
    if (accuracy < mediumScoreThreshold) {
      if (accuracy < badScoreThreshold) currentAudioForPlaying = soundFileBad;
      else currentAudioForPlaying = soundFileOkay;
    }
    playback();
  };

  const playAudio = async () => {
    const mainTitle = document.getElementById("main_title");
    if (mainTitle) mainTitle.innerHTML = "Generating sound...";
    playWithMozillaApi(currentText[0]);
    if (mainTitle) mainTitle.innerHTML = "Current Sound was played";
  };

  function playback() {
    const playSound = ctx.createBufferSource();
    playSound.buffer = currentAudioForPlaying;
    playSound.connect(ctx.destination);
    playSound.start(ctx.currentTime);
  }

  const playRecording = async (start = null, end = null) => {
    blockUI();
    try {
      if (start == null || end == null) {
        let endTimeInMs = Math.round(audioRecorded.duration * 1000);
        audioRecorded.addEventListener("ended", function () {
          audioRecorded.currentTime = 0;
          unblockUI();
          const mainTitle = document.getElementById("main_title");
          if (mainTitle) mainTitle.innerHTML = "Recorded Sound was played";
        });
        await audioRecorded.play();
      } else {
        audioRecorded.currentTime = start;
        audioRecorded.play();
        let durationInSeconds = end - start;
        let endTimeInMs = Math.round(durationInSeconds * 1000);
        setTimeout(function () {
          unblockUI();
          audioRecorded.pause();
          audioRecorded.currentTime = 0;
          const mainTitle = document.getElementById("main_title");
          if (mainTitle) mainTitle.innerHTML = "Recorded Sound was played";
        }, endTimeInMs);
      }
    } catch {
      UINotSupported();
    }
  };

  const playNativeAndRecordedWord = async (word_idx) => {
    if (isNativeSelectedForPlayback) playCurrentWord(word_idx);
    else playRecordedWord(word_idx);
    isNativeSelectedForPlayback = !isNativeSelectedForPlayback;
  };

  const stopRecording = async () => {
    isRecording = false;
    mediaRecorder.stop();
    const mainTitle = document.getElementById("main_title");
    if (mainTitle) mainTitle.innerHTML = "Processing audio...";
  };

  const playCurrentWord = async (word_idx) => {
    const mainTitle = document.getElementById("main_title");
    if (mainTitle) mainTitle.innerHTML = "Generating word...";
    playWithMozillaApi(currentText[0].split(" ")[word_idx]);
    if (mainTitle) mainTitle.innerHTML = "Word was played";
  };

  const playWithMozillaApi = (text) => {
    if (languageFound) {
      blockUI();
      if (voice_synth == null) changeLanguage(AILanguage);
      var utterThis = new SpeechSynthesisUtterance(text);
      utterThis.voice = voice_synth;
      utterThis.rate = 0.7;
      utterThis.onend = function (event) {
        unblockUI();
      };
      synth.speak(utterThis);
    } else {
      UINotSupported();
    }
  };

  const playRecordedWord = (word_idx) => {
    let wordStartTime = parseFloat(startTime.split(" ")[word_idx]);
    let wordEndTime = parseFloat(endTime.split(" ")[word_idx]);
    playRecording(wordStartTime, wordEndTime);
  };

  // Utils
  const convertBlobToBase64 = async (blob) => await blobToBase64(blob);

  const blobToBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  const wrapWordForPlayingLink = (
    word,
    word_idx,
    isFromRecording,
    word_accuracy_color
  ) =>
    isFromRecording
      ? `<a style="white-space:nowrap; color:${word_accuracy_color};" href="javascript:playRecordedWord(${word_idx})">${word}</a> `
      : `<a style="white-space:nowrap; color:${word_accuracy_color};" href="javascript:playCurrentWord(${word_idx})">${word}</a> `;

  const wrapWordForIndividualPlayback = (word, word_idx) =>
    `<a onmouseover="generateWordModal(${word_idx})" style="white-space:nowrap;" href="javascript:playNativeAndRecordedWord(${word_idx})">${word}</a> `;

  // Initialize server
  const initializeServer = async () => {
    let valid_response = false;
    const mainTitle = document.getElementById("main_title");
    if (mainTitle)
      mainTitle.innerHTML =
        "Initializing server, this may take up to 2 minutes...";
    let number_of_tries = 0;
    let maximum_number_of_tries = 4;
    while (!valid_response && number_of_tries <= maximum_number_of_tries) {
      try {
        const response = await fetch(
          apiMainPathSTS + "/GetAccuracyFromRecordedAudio",
          {
            method: "POST",
            body: JSON.stringify({
              title: "",
              base64Audio: "",
              language: AILanguage,
            }),
            headers: {
              "X-Api-Key": STScoreAPIKey,
              "Content-Type": "application/json",
            },
          }
        );
        if (response.ok) {
          valid_response = true;
          serverIsInitialized = true;
          serverWorking = true;
          if (mainTitle) mainTitle.innerHTML = page_title;
          unblockUI();
        } else console.error("Server response not OK:", response.status);
      } catch (e) {
        console.error("Server initialization error:", e);
        number_of_tries += 1;
        await new Promise((resolve) =>
          setTimeout(resolve, 1000 * number_of_tries)
        );
      }
    }
    if (!valid_response) {
      serverWorking = false;
      UIError("Server initialization failed after multiple attempts.");
    }
  };

  // Attach event listeners
  document.addEventListener("DOMContentLoaded", () => {
    unblockUI();
    const buttonNext = document.getElementById("buttonNext");
    if (buttonNext) buttonNext.addEventListener("click", getNextSample);

    const lengthCats = [
      document.getElementById("lengthCat1"),
      document.getElementById("lengthCat2"),
      document.getElementById("lengthCat3"),
      document.getElementById("lengthCat4"),
    ];
    lengthCats.forEach((cat) => {
      if (cat) cat.addEventListener("click", getNextSample);
    });

    const recordAudio = document.getElementById("recordAudio");
    if (recordAudio)
      recordAudio.addEventListener("click", updateRecordingState);

    const playSampleAudio = document.getElementById("playSampleAudio");
    if (playSampleAudio) playSampleAudio.addEventListener("click", playAudio);

    const playRecordedAudio = document.getElementById("playRecordedAudio");
    if (playRecordedAudio)
      playRecordedAudio.addEventListener("click", () => playRecording());

    const dropdown = document.querySelector(".dropdown-content");
    if (dropdown) {
      dropdown.addEventListener("click", function (e) {
        if (e.target.tagName === "A") {
          const language = e.target.getAttribute("data-language");
          const languageBox = document.getElementById("languageBox");
          if (languageBox) languageBox.textContent = e.target.textContent;
          changeLanguage(language, true);
        }
      });
    }

    const languageBox = document.getElementById("languageBox");
    if (languageBox) languageBox.textContent = "German";
  });
})();
