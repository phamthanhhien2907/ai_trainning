(function () {
  // Audio context initialization
  let mediaRecorder = undefined;
  let audioChunks = [];
  let audioBlob, stream;
  const ctx = new AudioContext();
  let currentScore = 0;
  let isRecording = false;
  let currentVocabulary = [];
  let currentIndex = 0;
  let lettersOfWordAreCorrect = [];
  let recordingTimeout = null; // To store the timeout ID

  // API related variables
  const apiMainPath = "http://127.0.0.1:3000";
  const STScoreAPIKey = "rll5QsTiv83nti99BW6uCmvs9BDVxSB39SVFceYb";

  // Scoring thresholds
  const badScoreThreshold = 30;
  const mediumScoreThreshold = 70;

  // Load vocabulary from Flask backend
  const loadVocabulary = async () => {
    try {
      const response = await fetch(apiMainPath + "/getVocabulary", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      currentVocabulary = data.filter(
        (item) => item.word && item.ipa && item.ending_sound
      );
      if (currentVocabulary.length === 0)
        throw new Error("No valid vocabulary data received");
      updateDisplay();
    } catch (error) {
      console.error("Error loading vocabulary:", error);
      // Fallback vocabulary in case the API fails
      currentVocabulary = [
        { word: "web", ipa: "wɛb", ending_sound: "b" },
        { word: "wet", ipa: "wɛt", ending_sound: "t" },
        { word: "pin", ipa: "pɪn", ending_sound: "n" },
      ];
      displayError("Failed to load vocabulary. Using fallback data.");
      updateDisplay();
    }
  };

  // UI control functions
  const blockUI = () => {
    const elements = ["recordAudio", "playSampleAudio", "buttonNext"];
    elements.forEach((id) => {
      const element = document.getElementById(id);
      if (element) element.classList.add("disabled");
    });
  };

  const unblockUI = () => {
    const elements = ["recordAudio", "playSampleAudio", "buttonNext"];
    elements.forEach((id) => {
      const element = document.getElementById(id);
      if (element) element.classList.remove("disabled");
    });
  };

  const displayError = (message) => {
    blockUI();
    const mainTitle = document.getElementById("main_title");
    const originalScript = document.getElementById("original_script");
    if (mainTitle) mainTitle.innerHTML = "Error";
    if (originalScript) originalScript.innerHTML = message;
    setTimeout(() => {
      if (mainTitle) mainTitle.innerHTML = "Ending Sounds Trainer";
      if (originalScript && !currentVocabulary[currentIndex]) {
        originalScript.innerHTML = "Click record to start.";
      }
      unblockUI();
    }, 3000);
  };

  const updateDisplay = () => {
    const currentWord = currentVocabulary[currentIndex];
    if (currentWord) {
      const mainTitle = document.getElementById("main_title");
      const originalScript = document.getElementById("original_script");
      const ipaScript = document.getElementById("ipa_script");
      const recordedIpaScript = document.getElementById("recorded_ipa_script");
      const pronunciationAccuracy = document.getElementById(
        "pronunciation_accuracy"
      );
      const singleWordIpa = document.getElementById("single_word_ipa");

      if (mainTitle) mainTitle.innerHTML = "Ending Sounds Trainer";
      if (originalScript) originalScript.innerHTML = currentWord.word;
      if (ipaScript) ipaScript.innerHTML = `/ ${currentWord.ipa} /`;
      if (recordedIpaScript) recordedIpaScript.innerHTML = "";
      if (pronunciationAccuracy) pronunciationAccuracy.innerHTML = "";
      if (singleWordIpa) singleWordIpa.innerHTML = "Reference | Spoken";
    } else {
      displayError("No vocabulary available. Please refresh the page.");
    }
  };

  // Audio recording
  const mediaStreamConstraints = {
    audio: {
      channelCount: 1,
      sampleRate: 48000,
    },
  };

  const startMediaDevice = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia(
        mediaStreamConstraints
      );
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
      mediaRecorder.onstop = processRecording;
      unblockUI();
      const mainTitle = document.getElementById("main_title");
      if (mainTitle) mainTitle.innerHTML = "Ending Sounds Trainer - Ready";
    } catch (e) {
      console.error("Media device error:", e);
      displayError("Microphone access denied. Click the mic to retry.");
      const recordAudio = document.getElementById("recordAudio");
      if (recordAudio) recordAudio.classList.remove("disabled");
    }
  };

  const recordSample = async () => {
    if (!mediaRecorder || !stream) {
      await startMediaDevice();
      if (!mediaRecorder) {
        displayError("Failed to access microphone. Please refresh the page.");
        return;
      }
    }
    if (!isRecording) {
      audioChunks = [];
      isRecording = true;
      const mainTitle = document.getElementById("main_title");
      const recordIcon = document.getElementById("recordIcon");
      if (mainTitle) mainTitle.innerHTML = "Recording...";
      if (recordIcon) recordIcon.innerHTML = "pause_presentation";
      blockUI();
      try {
        mediaRecorder.start();
        recordingTimeout = setTimeout(() => {
          if (isRecording) {
            stopRecording();
          }
        }, 10000);
      } catch (error) {
        console.error("Error starting recording:", error);
        displayError("Failed to start recording. Please try again.");
        isRecording = false;
        unblockUI();
      }
    } else {
      clearTimeout(recordingTimeout);
      await stopRecording();
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      isRecording = false;
      mediaRecorder.stop();
      clearTimeout(recordingTimeout);
      const mainTitle = document.getElementById("main_title");
      const recordIcon = document.getElementById("recordIcon");
      if (mainTitle) mainTitle.innerHTML = "Processing audio...";
      if (recordIcon) recordIcon.innerHTML = "mic";
    }
  };

  const processRecording = async () => {
    if (audioChunks.length === 0) {
      displayError("No audio data recorded. Please try again.");
      return;
    }
    audioBlob = new Blob(audioChunks, { type: "audio/ogg;" });
    const audioBase64 = await convertBlobToBase64(audioBlob);
    console.log("Base64 string sent length:", audioBase64.length);
    console.log("Base64 string sent:", audioBase64.substring(0, 100) + "...");

    try {
      const currentWord = currentVocabulary[currentIndex];
      const response = await fetch(
        apiMainPath + "/GetAccuracyFromRecordedAudio",
        {
          method: "POST",
          body: JSON.stringify({
            title: currentWord.word,
            base64Audio: audioBase64,
            language: "en",
          }),
          headers: {
            "X-Api-Key": STScoreAPIKey,
            "Content-Type": "application/json",
          },
          signal: AbortSignal.timeout(15000),
        }
      );

      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      console.log("API response from /GetAccuracyFromRecordedAudio:", data);
      if (data.error) throw new Error(data.error);

      const accuracy = parseFloat(data.pronunciation_accuracy) || 0;
      updateScore(accuracy);
      displayFeedback(data, currentWord);
      unblockUI();
    } catch (e) {
      console.error("Audio processing error:", e);
      displayError(`Failed to process audio: ${e.message}`);
    }
  };

  const playSampleAudio = async () => {
    try {
      const currentWord = currentVocabulary[currentIndex];
      const response = await fetch(apiMainPath + "/getAudioFromText", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: currentWord.word,
          language: "en",
        }),
      });

      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      console.log("API response from /getAudioFromText:", data);
      if (data.audio_url) {
        const audio = new Audio(data.audio_url);
        audio.play().catch((error) => {
          console.error("Error playing audio:", error);
          displayError("Failed to play audio.");
        });
      } else {
        throw new Error("No audio URL returned from server");
      }
    } catch (error) {
      console.error("Error generating audio:", error);
      displayError(`Failed to generate audio for the word: ${error.message}`);
    }
  };

  const updateScore = (accuracy) => {
    currentScore += accuracy;
    currentScore = Math.round(currentScore);
    const sectionAccuracy = document.getElementById("section_accuracy");
    if (sectionAccuracy) {
      sectionAccuracy.innerHTML = `| Score: ${currentScore} - (${
        currentIndex + 1
      }/${currentVocabulary.length})`;
    }
  };

  const displayFeedback = (data, currentWord) => {
    const recordedIpaScript = document.getElementById("recorded_ipa_script");
    const pronunciationAccuracy = document.getElementById(
      "pronunciation_accuracy"
    );
    const singleWordIpa = document.getElementById("single_word_ipa");
    const originalScript = document.getElementById("original_script");

    if (recordedIpaScript)
      recordedIpaScript.innerHTML = `/ ${data.ipa_transcript || ""} /`;
    if (pronunciationAccuracy)
      pronunciationAccuracy.innerHTML = `${data.pronunciation_accuracy}%`;
    if (singleWordIpa) {
      singleWordIpa.innerHTML = `Reference: / ${
        currentWord.ipa
      } / | Spoken: / ${data.ipa_transcript || ""} /`;
    }

    const normalizeIPA = (ipa) =>
      ipa
        .replace(/\/|\[|\]/g, "")
        .trim()
        .toLowerCase();
    const expectedEnding = normalizeIPA(currentWord.ending_sound);
    const spokenEnding = data.ipa_transcript
      ? normalizeIPA(data.ipa_transcript.split(" ").pop())
      : "";
    const isEndingCorrect = expectedEnding === spokenEnding;
    const accuracy = parseFloat(data.pronunciation_accuracy) || 0;
    const color = isEndingCorrect
      ? "green"
      : accuracy >= mediumScoreThreshold
      ? "orange"
      : "red";

    if (originalScript) {
      originalScript.innerHTML = `<span style="color: ${color}">${currentWord.word}</span> (Ending: ${currentWord.ending_sound})`;
    }
  };

  const nextSample = () => {
    if (currentIndex < currentVocabulary.length - 1) {
      currentIndex++;
      updateDisplay();
    } else {
      const finalMessage = `Completed all vocabulary! Final Score: ${currentScore}`;
      alert(finalMessage);
      const mainTitle = document.getElementById("main_title");
      const originalScript = document.getElementById("original_script");
      if (mainTitle) mainTitle.innerHTML = "Completed!";
      if (originalScript) originalScript.innerHTML = finalMessage;
      blockUI();
    }
  };

  const convertBlobToBase64 = async (blob) => await blobToBase64(blob);

  const blobToBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onload = () => {
        let base64String = reader.result?.split(",")[1] || "";
        console.log("Raw base64 length before padding:", base64String.length);
        const paddingLength = (4 - (base64String.length % 4)) % 4;
        base64String += "=".repeat(paddingLength);
        console.log("Padded base64 length:", base64String.length);
        if (base64String.length % 4 !== 0) {
          console.error(
            "Base64 string length is not a multiple of 4 after padding:",
            base64String.length
          );
        }
        resolve(base64String);
      };
      reader.onerror = (error) => reject(error);
    });

  document.addEventListener("DOMContentLoaded", () => {
    startMediaDevice();
    loadVocabulary();

    const recordAudio = document.getElementById("recordAudio");
    const playSampleAudioBtn = document.getElementById("playSampleAudio");
    const buttonNext = document.getElementById("buttonNext");

    if (recordAudio) recordAudio.addEventListener("click", recordSample);
    if (playSampleAudioBtn)
      playSampleAudioBtn.addEventListener("click", playSampleAudio);
    if (buttonNext) buttonNext.addEventListener("click", nextSample);
  });
})();
