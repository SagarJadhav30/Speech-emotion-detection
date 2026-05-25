import React, { useState, useRef } from 'react';
import 'tailwindcss/tailwind.css';

function App() {
  const [recording, setRecording] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictedEmotion, setPredictedEmotion] = useState('');
  const [mode, setMode] = useState(null); // 'upload' or 'record'
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isButtonDisabled, setIsButtonDisabled] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setError('');
  };

  const playBeep = () => {
    const beep = new Audio('/beepsound.wav');
    beep.play();
  };

  const handleStartRecording = () => {
    playBeep();
    setTimeout(() => {
      setRecording(true);
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          mediaRecorderRef.current = new MediaRecorder(stream);
          mediaRecorderRef.current.ondataavailable = event => {
            audioChunksRef.current.push(event.data);
          };
          mediaRecorderRef.current.start();
        })
        .catch(error => {
          console.error('Error accessing microphone:', error);
        });
    }, 500); // Delay to allow beep sound to play
  };

  const handleStopRecording = () => {
    playBeep();
    setTimeout(() => {
      setRecording(false);
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        audioChunksRef.current = [];
        setRecordedBlob(audioBlob);
      };
    }, 500); // Delay to allow beep sound to play
  };

  const handleUploadRecording = async () => {
    setIsButtonDisabled(true);
    const formData = new FormData();
    formData.append('audio', recordedBlob, 'recording.wav'); // Ensure the file has a proper name and extension

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPredictedEmotion(data.predicted_emotion);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false);
      setIsButtonDisabled(false);
    }
  };

  const handleSubmitFile = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please upload an audio file.');
      return;
    }
    setIsButtonDisabled(true);
    const formData = new FormData();
    formData.append('audio', selectedFile);

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPredictedEmotion(data.predicted_emotion);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false);
      setIsButtonDisabled(false);
    }
  };

  const getBackgroundClass = () => {
    if (predictedEmotion.includes('happy')) return 'bg-gradient-to-br from-yellow-400 via-orange-500 to-red-500';
    if (predictedEmotion.includes('sad')) return 'bg-gradient-to-br from-blue-400 via-blue-600 to-indigo-800';
    if (predictedEmotion.includes('angry')) return 'bg-gradient-to-br from-red-400 via-red-600 to-red-800';
    if (predictedEmotion.includes('fearful')) return 'bg-gradient-to-br from-purple-400 via-purple-600 to-purple-800';
    if (predictedEmotion.includes('disgust')) return 'bg-gradient-to-br from-green-400 via-green-600 to-green-800';
    if (predictedEmotion.includes('surprised')) return 'bg-gradient-to-br from-pink-400 via-pink-600 to-pink-800';
    return 'bg-gray-900';
  };

  const getEmoji = () => {
    const emoji = predictedEmotion.split(' ')[1];
    return emoji ? emoji : '';
  };

  const handleBack = () => {
    window.location.reload();
  };

  return (
    <div className={`min-h-screen flex items-center justify-center transition-all duration-500 ease-out ${predictedEmotion ? getBackgroundClass() : 'bg-gradient-to-br from-gray-900 via-gray-700 to-gray-300'}`}>
      <div className="bg-white/70 backdrop-blur-md p-8 rounded-xl shadow-2xl w-full max-w-md">
        <h1 className="text-3xl font-bold mb-6 text-gray-900 text-center">Speech Emotion Detection</h1>
        {!mode && (
          <div className="space-y-4">
            <button 
              onClick={() => setMode('upload')}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
            >
              Upload Audio File
            </button>
            <button 
              onClick={() => setMode('record')}
              className="w-full bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
            >
              Record Audio
            </button>
          </div>
        )}
        {mode === 'upload' && (
          <form onSubmit={handleSubmitFile} className="space-y-4">
            <input 
              type="file" 
              accept="audio/*" 
              onChange={handleFileChange} 
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <button 
              type="submit" 
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
              disabled={isButtonDisabled}
            >
              Upload and Predict
            </button>
            <button 
              onClick={handleBack}
              className="w-full bg-gray-600 hover:bg-gray-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
            >
              Back
            </button>
          </form>
        )}
        {mode === 'record' && (
          <div className="space-y-4">
            {!recording && !recordedBlob && (
              <>
                <button 
                  onClick={handleStartRecording} 
                  className="w-full bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
                >
                  Start Recording
                </button>
                <button 
                  onClick={handleBack}
                  className="w-full bg-gray-600 hover:bg-gray-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
                >
                  Back
                </button>
              </>
            )}
            {recording && (
              <button 
                onClick={handleStopRecording} 
                className="w-full bg-red-600 hover:bg-red-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
              >
                Stop Recording
              </button>
            )}
            {recordedBlob && (
              <>
                <button 
                  onClick={handleUploadRecording}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
                  disabled={isButtonDisabled}
                >
                  Upload Recording
                </button>
                <button 
                  onClick={handleBack}
                  className="w-full bg-gray-600 hover:bg-gray-700 text-white py-3 px-6 rounded-lg font-semibold uppercase tracking-wide shadow-lg transition transform hover:scale-105"
                >
                  Back
                </button>
              </>
            )}
          </div>
        )}
        {loading && <p className="mt-4 text-lg font-semibold text-gray-900">Loading...</p>}
        {predictedEmotion && !loading && (
          <div className="flex flex-col items-center space-y-6">
            <div className="text-8xl animate-shake">{getEmoji()}</div>
            <p className="text-lg font-semibold text-gray-900">Detected Emotion: {predictedEmotion}</p>
            <button 
              onClick={handleBack} 
              className="bg-gray-500 text-white py-2 px-4 rounded hover:bg-gray-600"
            >
              Back
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
