# MoodMelody 🎶
Understand Your Mood, Find Your Music
MoodMelody is an AI-powered music recommender that bridges the gap between your emotions and the perfect soundtrack. Built for those moments when you just can't decide what to play, the app uses Natural Language Processing (NLP) and a lightweight BiLSTM classifier to understand your feelings and suggest music to match.

# How It Works: A Technical Overview 
The core of MoodMelody is a simple yet powerful workflow: Text → Mood → Music.

1. Text to Mood Prediction
You start by typing a thought or feeling into the app. The system then processes this text through a series of steps:

# Preprocessing: 
The text is cleaned, lowercased, and tokenized.

# Embedding: 
Pre-trained GloVe word embeddings convert your words into dense numerical vectors, helping the model understand their meaning.

# Classification: 
A trained BiLSTM (Bidirectional Long Short-Term Memory) model analyzes these vectors to predict the underlying mood or emotion. This model is intentionally small to ensure a fast, "snappy" user experience.

#  Mood to Music Recommendation
Once a mood is identified, the app connects to the Spotify Web API using the Spotipy library.

Each mood is mapped to specific search queries or audio feature hints.

The app fetches relevant tracks or playlists that align with that mood, providing instant recommendations.

This approach uses a combination of semantic understanding and API curation to deliver relevant music.

# Spotify Integration
The app handles Spotify authentication using OAuth, ensuring a secure and personalized experience.

When you log in, Spotify issues an access token tied to your account, not the app developer's.

This allows the app to control playback directly on your Spotify account and devices, providing a seamless user experience. The app’s client credentials only identify it as a valid application; all playback actions are tied to your personal account.

# Key Features
Emotion Detection: Predicts emotions from any free-form text input.

Accurate Prediction: Uses a BiLSTM classifier with pre-trained GloVe embeddings for high-quality semantic understanding.

Smart Recommendations: Maps detected emotions to curated Spotify playlists and tracks.

Seamless Integration: Utilizes the Spotify API with secure OAuth authentication for personalized playback.

Lightweight and Fast: The model and deployment on Streamlit ensure the app is responsive and easy to use.

