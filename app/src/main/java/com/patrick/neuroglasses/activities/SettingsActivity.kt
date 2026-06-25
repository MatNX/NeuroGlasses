package com.patrick.neuroglasses.activities

import android.content.Context
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.patrick.neuroglasses.R

class SettingsActivity : AppCompatActivity() {

    // UI Components
    private lateinit var apiBaseUrlEditText: EditText
    private lateinit var apiTokenEditText: EditText
    private lateinit var apiTimeoutEditText: EditText
    private lateinit var systemPromptEditText: EditText
    private lateinit var vlmModelEditText: EditText
    private lateinit var vlmMaxTokensEditText: EditText
    private lateinit var asrModelEditText: EditText
    private lateinit var ttsModelEditText: EditText
    private lateinit var ttsVoiceEditText: EditText
    private lateinit var mapTilerApiKeyEditText: EditText
    private lateinit var saveButton: Button
    private lateinit var resetButton: Button

    companion object {
        // SharedPreferences keys
        const val PREFS_NAME = "GroqSettings"
        const val KEY_API_BASE_URL = "api_base_url"
        const val KEY_API_TOKEN = "api_token"
        const val KEY_API_TIMEOUT = "api_timeout"
        const val KEY_SYSTEM_PROMPT = "system_prompt"
        const val KEY_VLM_MODEL = "vlm_model"
        const val KEY_VLM_MAX_TOKENS = "vlm_max_tokens"
        const val KEY_ASR_MODEL = "asr_model"
        const val KEY_TTS_MODEL = "tts_model"
        const val KEY_TTS_VOICE = "tts_voice"
        const val KEY_MAPTILER_API_KEY = "maptiler_api_key"

        // Default values
        const val DEFAULT_API_BASE_URL = "https://api.groq.com/openai/v1"
        const val DEFAULT_API_TOKEN = ""
        const val DEFAULT_API_TIMEOUT = 15
        const val DEFAULT_SYSTEM_PROMPT = "Du bist ein deutschsprachiger KI-Assistent für Rokid-AR-Brillen in Österreich. Lies die Absicht des Nutzers mit. Antworte kurz, natürlich und freihändig nutzbar. Nutze sichere Tools proaktiv für Anrufe, SMS, Navigation über MapTiler, Wetter, Websuche, Erinnerungen, Kalender, E-Mail, Apps, Teilen, Akku und Fotos, damit der Nutzer das Telefon nach der Einrichtung möglichst nicht mehr ansehen muss. Wenn der Nutzer etwas Sichtbares meint, z. B. Finger, Text, Objekte, Farben oder \"was sehe ich\", verwende ein Foto als Kontext. Wenn der Nutzer sich verabschiedet, beende das Gespräch freundlich. Frage nur nach, wenn eine Pflichtangabe fehlt oder eine Aktion unsicher wäre. Führe destruktive oder riskante Aktionen nicht ohne eindeutige Bestätigung aus."
        const val DEFAULT_VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
        const val DEFAULT_VLM_MAX_TOKENS = 1024
        const val DEFAULT_ASR_MODEL = "whisper-large-v3"
        const val DEFAULT_TTS_MODEL = "playai-tts"
        const val DEFAULT_TTS_VOICE = "Arista-PlayAI"
        const val DEFAULT_MAPTILER_API_KEY = ""

        // Helper functions to get configuration values
        fun getApiBaseUrl(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_API_BASE_URL, DEFAULT_API_BASE_URL) ?: DEFAULT_API_BASE_URL
        }

        fun getApiToken(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_API_TOKEN, DEFAULT_API_TOKEN) ?: DEFAULT_API_TOKEN
        }

        fun getApiTimeout(context: Context): Int {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getInt(KEY_API_TIMEOUT, DEFAULT_API_TIMEOUT)
        }

        fun getSystemPrompt(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT) ?: DEFAULT_SYSTEM_PROMPT
        }

        fun getVlmModel(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_VLM_MODEL, DEFAULT_VLM_MODEL) ?: DEFAULT_VLM_MODEL
        }

        fun getVlmMaxTokens(context: Context): Int {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getInt(KEY_VLM_MAX_TOKENS, DEFAULT_VLM_MAX_TOKENS)
        }

        fun getAsrModel(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_ASR_MODEL, DEFAULT_ASR_MODEL) ?: DEFAULT_ASR_MODEL
        }

        fun getTtsModel(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_TTS_MODEL, DEFAULT_TTS_MODEL) ?: DEFAULT_TTS_MODEL
        }

        fun getTtsVoice(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_TTS_VOICE, DEFAULT_TTS_VOICE) ?: DEFAULT_TTS_VOICE
        }

        fun getMapTilerApiKey(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getString(KEY_MAPTILER_API_KEY, DEFAULT_MAPTILER_API_KEY) ?: DEFAULT_MAPTILER_API_KEY
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        // Initialize UI components
        apiBaseUrlEditText = findViewById(R.id.apiBaseUrlEditText)
        apiTokenEditText = findViewById(R.id.apiTokenEditText)
        apiTimeoutEditText = findViewById(R.id.apiTimeoutEditText)
        systemPromptEditText = findViewById(R.id.systemPromptEditText)
        vlmModelEditText = findViewById(R.id.vlmModelEditText)
        vlmMaxTokensEditText = findViewById(R.id.vlmMaxTokensEditText)
        asrModelEditText = findViewById(R.id.asrModelEditText)
        ttsModelEditText = findViewById(R.id.ttsModelEditText)
        ttsVoiceEditText = findViewById(R.id.ttsVoiceEditText)
        mapTilerApiKeyEditText = findViewById(R.id.mapTilerApiKeyEditText)
        saveButton = findViewById(R.id.saveButton)
        resetButton = findViewById(R.id.resetButton)

        // Load current settings
        loadSettings()

        // Set up button listeners
        saveButton.setOnClickListener {
            saveSettings()
        }

        resetButton.setOnClickListener {
            resetToDefaults()
        }
    }

    private fun loadSettings() {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)

        apiBaseUrlEditText.setText(prefs.getString(KEY_API_BASE_URL, DEFAULT_API_BASE_URL))
        apiTokenEditText.setText(prefs.getString(KEY_API_TOKEN, DEFAULT_API_TOKEN))
        apiTimeoutEditText.setText(prefs.getInt(KEY_API_TIMEOUT, DEFAULT_API_TIMEOUT).toString())
        systemPromptEditText.setText(prefs.getString(KEY_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT))
        vlmModelEditText.setText(prefs.getString(KEY_VLM_MODEL, DEFAULT_VLM_MODEL))
        vlmMaxTokensEditText.setText(prefs.getInt(KEY_VLM_MAX_TOKENS, DEFAULT_VLM_MAX_TOKENS).toString())
        asrModelEditText.setText(prefs.getString(KEY_ASR_MODEL, DEFAULT_ASR_MODEL))
        ttsModelEditText.setText(prefs.getString(KEY_TTS_MODEL, DEFAULT_TTS_MODEL))
        ttsVoiceEditText.setText(prefs.getString(KEY_TTS_VOICE, DEFAULT_TTS_VOICE))
        mapTilerApiKeyEditText.setText(prefs.getString(KEY_MAPTILER_API_KEY, DEFAULT_MAPTILER_API_KEY))
    }

    private fun saveSettings() {
        try {
            val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            val editor = prefs.edit()

            // Validate and save API base URL
            val apiBaseUrl = apiBaseUrlEditText.text.toString().trim()
            if (apiBaseUrl.isEmpty()) {
                Toast.makeText(this, "Groq-API-Basis-URL darf nicht leer sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putString(KEY_API_BASE_URL, apiBaseUrl)

            // Save API token
            val apiToken = apiTokenEditText.text.toString().trim()
            editor.putString(KEY_API_TOKEN, apiToken)

            // Validate and save API timeout
            val apiTimeoutStr = apiTimeoutEditText.text.toString().trim()
            val apiTimeout = apiTimeoutStr.toIntOrNull()
            if (apiTimeout == null || apiTimeout <= 0) {
                Toast.makeText(this, "API-Timeout muss eine positive Zahl in Sekunden sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putInt(KEY_API_TIMEOUT, apiTimeout)

            // Validate and save system prompt
            val systemPrompt = systemPromptEditText.text.toString().trim()
            if (systemPrompt.isEmpty()) {
                Toast.makeText(this, "System-Prompt darf nicht leer sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putString(KEY_SYSTEM_PROMPT, systemPrompt)

            // Validate and save VLM model
            val vlmModel = vlmModelEditText.text.toString().trim()
            if (vlmModel.isEmpty()) {
                Toast.makeText(this, "VLM-Modell darf nicht leer sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putString(KEY_VLM_MODEL, vlmModel)

            // Validate and save VLM max tokens
            val vlmMaxTokensStr = vlmMaxTokensEditText.text.toString().trim()
            val vlmMaxTokens = vlmMaxTokensStr.toIntOrNull()
            if (vlmMaxTokens == null || vlmMaxTokens <= 0) {
                Toast.makeText(this, "Maximale VLM-Tokens müssen eine positive Zahl sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putInt(KEY_VLM_MAX_TOKENS, vlmMaxTokens)

            // Validate and save ASR model
            val asrModel = asrModelEditText.text.toString().trim()
            if (asrModel.isEmpty()) {
                Toast.makeText(this, "ASR-Modell darf nicht leer sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putString(KEY_ASR_MODEL, asrModel)

            // Validate and save TTS model
            val ttsModel = ttsModelEditText.text.toString().trim()
            if (ttsModel.isEmpty()) {
                Toast.makeText(this, "TTS-Modell darf nicht leer sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putString(KEY_TTS_MODEL, ttsModel)

            // Validate and save TTS voice
            val ttsVoice = ttsVoiceEditText.text.toString().trim()
            if (ttsVoice.isEmpty()) {
                Toast.makeText(this, "TTS-Stimme darf nicht leer sein", Toast.LENGTH_SHORT).show()
                return
            }
            editor.putString(KEY_TTS_VOICE, ttsVoice)

            editor.putString(KEY_MAPTILER_API_KEY, mapTilerApiKeyEditText.text.toString().trim())

            // Apply changes
            editor.apply()

            Toast.makeText(this, "Einstellungen gespeichert", Toast.LENGTH_SHORT).show()
            finish()
        } catch (e: Exception) {
            Toast.makeText(this, "Fehler beim Speichern: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun resetToDefaults() {
        apiBaseUrlEditText.setText(DEFAULT_API_BASE_URL)
        apiTokenEditText.setText(DEFAULT_API_TOKEN)
        apiTimeoutEditText.setText(DEFAULT_API_TIMEOUT.toString())
        systemPromptEditText.setText(DEFAULT_SYSTEM_PROMPT)
        vlmModelEditText.setText(DEFAULT_VLM_MODEL)
        vlmMaxTokensEditText.setText(DEFAULT_VLM_MAX_TOKENS.toString())
        asrModelEditText.setText(DEFAULT_ASR_MODEL)
        ttsModelEditText.setText(DEFAULT_TTS_MODEL)
        ttsVoiceEditText.setText(DEFAULT_TTS_VOICE)
        mapTilerApiKeyEditText.setText(DEFAULT_MAPTILER_API_KEY)

        Toast.makeText(this, "Auf Standardwerte zurückgesetzt", Toast.LENGTH_SHORT).show()
    }
}
