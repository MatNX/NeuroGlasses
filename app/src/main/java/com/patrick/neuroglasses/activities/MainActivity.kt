package com.patrick.neuroglasses.activities

import android.app.Activity
import android.content.ComponentName
import android.content.Intent
import android.net.Uri
import android.util.Log
import android.bluetooth.BluetoothDevice
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.tabs.TabLayout
import com.patrick.neuroglasses.R
import com.patrick.neuroglasses.helpers.AppPermissions
import com.patrick.neuroglasses.helpers.BluetoothHelper
import com.patrick.neuroglasses.helpers.OpenAIHelper
import com.patrick.neuroglasses.helpers.PersistentConversationSummary
import com.patrick.neuroglasses.helpers.RokidHostConnection
import com.patrick.neuroglasses.services.RokidConnectionService
import java.text.DateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private val appTag = "NeuroGlasses"

    companion object {
        // Request Code
        const val REQUEST_CODE_PERMISSIONS = 100
        const val REQUEST_CODE_ROKID_AUTHORIZATION = 4027

        private const val ROKID_AUTH_ACTIVITY_CLASS =
            "com.rokid.sprite.aiapp.externalapp.auth.AuthorizationActivity"
        private const val ROKID_AUTH_ACTION =
            "com.rokid.sprite.aiapp.externalapp.AUTHORIZATION"
        private const val ROKID_AUTH_TOKEN_EXTRA = "auth_token"
        private const val ROKID_AUTH_RESULT_EXTRA = "auth_result"
        private const val ROKID_AUTH_RESULT_SUCCESS = 2001
        private const val ROKID_AUTH_RESULT_CANCEL = 2003
    }

    private lateinit var bluetoothHelper: BluetoothHelper
    private lateinit var deviceListView: ListView
    private lateinit var statusTextView: TextView
    private lateinit var navigateButton: Button
    private lateinit var testUIButton: Button
    private lateinit var scanButton: Button
    private lateinit var mainTabLayout: TabLayout
    private lateinit var devicesTabContent: View
    private lateinit var conversationsTabContent: View
    private lateinit var conversationModeTextView: TextView
    private lateinit var conversationInfoTextView: TextView
    private lateinit var conversationListView: ListView
    private lateinit var newConversationButton: Button
    private lateinit var leaveConversationButton: Button
    private lateinit var renameConversationButton: Button
    private lateinit var deleteConversationButton: Button
    private lateinit var refreshConversationsButton: Button
    private lateinit var deviceListAdapter: ArrayAdapter<String>
    private lateinit var conversationListAdapter: ArrayAdapter<String>
    private lateinit var openAIHelper: OpenAIHelper
    private val deviceList = mutableListOf<String>()
    private val deviceMap = mutableMapOf<String, BluetoothDevice>()
    private val conversationDisplayList = mutableListOf<String>()
    private val conversationSummaries = mutableListOf<PersistentConversationSummary>()
    private var selectedConversationId: String? = null
    private var pendingAuthorizationDeviceName: String? = null
    private var isWaitingForRokidAuthorization = false
    private var isGlassesConnected = false

    private data class RokidHostApp(
        val displayName: String,
        val packageName: String
    )

    private val rokidAuthorizationHostApps = listOf(
        RokidHostApp("Hi Rokid Global", "com.rokid.sprite.global.aiapp"),
        RokidHostApp("Rokid AI CN", "com.rokid.sprite.aiapp")
    )

    // Helper method to check if Bluetooth permissions are granted
    private fun hasBluetoothPermissions(): Boolean {
        return AppPermissions.hasBluetoothPermissions(this)
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Initialize UI components
        deviceListView = findViewById(R.id.deviceListView)
        statusTextView = findViewById(R.id.statusTextView)
        navigateButton = findViewById(R.id.navigateButton)
        testUIButton = findViewById(R.id.testUIButton)
        scanButton = findViewById(R.id.scanButton)
        mainTabLayout = findViewById(R.id.mainTabLayout)
        devicesTabContent = findViewById(R.id.devicesTabContent)
        conversationsTabContent = findViewById(R.id.conversationsTabContent)
        conversationModeTextView = findViewById(R.id.conversationModeTextView)
        conversationInfoTextView = findViewById(R.id.conversationInfoTextView)
        conversationListView = findViewById(R.id.conversationListView)
        newConversationButton = findViewById(R.id.newConversationButton)
        leaveConversationButton = findViewById(R.id.leaveConversationButton)
        renameConversationButton = findViewById(R.id.renameConversationButton)
        deleteConversationButton = findViewById(R.id.deleteConversationButton)
        refreshConversationsButton = findViewById(R.id.refreshConversationsButton)
        openAIHelper = OpenAIHelper(this, appTag)

        // Setup device list adapter
        deviceListAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, deviceList)
        deviceListView.adapter = deviceListAdapter
        conversationListAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, conversationDisplayList)
        conversationListView.adapter = conversationListAdapter
        setupTabs()
        setupConversationControls()
        RokidHostConnection.setConnectionListener(object : RokidHostConnection.ConnectionListener {
            override fun onConnectionChanged(cxrConnected: Boolean, glassesConnected: Boolean) {
                runOnUiThread {
                    updateRokidHostConnectionStatus(cxrConnected, glassesConnected)
                }
            }
        })

        // Setup device list click listener
        deviceListView.setOnItemClickListener { _, _, position, _ ->
            val deviceName = deviceList[position]
            val device = deviceMap[deviceName]
            device?.let {
                showConnectDialog(it)
            }
        }

        // Setup navigate button
        navigateButton.setOnClickListener {
            val intent = Intent(this, AITestActivity::class.java)
            startActivity(intent)
        }

        // Setup settings button
        testUIButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
        }

        // Setup scan button
        scanButton.setOnClickListener {
            if (hasBluetoothPermissions()) {
                try {
                    bluetoothHelper.startScan()
                    statusTextView.setText(R.string.status_scanning)
                } catch (e: SecurityException) {
                    Log.e(appTag, "SecurityException when starting scan", e)
                    Toast.makeText(this, R.string.toast_bluetooth_permission_not_granted, Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, R.string.toast_bluetooth_permission_not_granted, Toast.LENGTH_SHORT).show()
            }
        }

        // Initialize BluetoothHelper
        bluetoothHelper = BluetoothHelper(
            context = this,
            initStatus = { status ->
                runOnUiThread {
                    when (status) {
                        BluetoothHelper.Companion.INITSTATUS.NotStart -> {
                            statusTextView.setText(R.string.status_bluetooth_not_started)
                        }
                        BluetoothHelper.Companion.INITSTATUS.INITING -> {
                            statusTextView.setText(R.string.status_bluetooth_initializing)
                        }
                        BluetoothHelper.Companion.INITSTATUS.INIT_END -> {
                            statusTextView.setText(R.string.status_bluetooth_initialized)
                        }
                    }
                }
            },
            deviceFound = {
                runOnUiThread {
                    updateDeviceList()
                }
            }
        )

        // Check permissions
        bluetoothHelper.checkPermissions()
        connectSavedRokidHostSession()
    }

    private fun setupTabs() {
        mainTabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                showMainTab(tab.position)
            }

            override fun onTabUnselected(tab: TabLayout.Tab) = Unit
            override fun onTabReselected(tab: TabLayout.Tab) {
                if (tab.position == 1) refreshConversationList()
            }
        })
        showMainTab(mainTabLayout.selectedTabPosition.coerceAtLeast(0))
    }

    private fun showMainTab(position: Int) {
        devicesTabContent.visibility = if (position == 0) View.VISIBLE else View.GONE
        conversationsTabContent.visibility = if (position == 1) View.VISIBLE else View.GONE
        if (position == 1) refreshConversationList()
    }

    private fun setupConversationControls() {
        conversationListView.setOnItemClickListener { _, _, position, _ ->
            val summary = conversationSummaries.getOrNull(position) ?: return@setOnItemClickListener
            selectedConversationId = summary.id
            if (openAIHelper.enterPersistentConversation(summary.id)) {
                Toast.makeText(this, "Persistente Konversation aktiv: ${summary.title}", Toast.LENGTH_SHORT).show()
                refreshConversationList()
            }
        }

        conversationListView.setOnItemLongClickListener { _, _, position, _ ->
            val summary = conversationSummaries.getOrNull(position) ?: return@setOnItemLongClickListener true
            selectedConversationId = summary.id
            promptRenameConversation(summary)
            true
        }

        newConversationButton.setOnClickListener {
            promptConversationTitle(
                title = "Neue persistente Konversation",
                prefill = "",
                allowBlank = true,
                positiveLabel = "Erstellen"
            ) { title ->
                val created = openAIHelper.createPersistentConversation(title.takeIf { it.isNotBlank() })
                selectedConversationId = created.id
                Toast.makeText(this, "Persistente Konversation erstellt", Toast.LENGTH_SHORT).show()
                refreshConversationList()
            }
        }

        leaveConversationButton.setOnClickListener {
            openAIHelper.leavePersistentConversation()
            selectedConversationId = null
            Toast.makeText(this, "Hauptkonversation ist wieder ephemer", Toast.LENGTH_SHORT).show()
            refreshConversationList()
        }

        renameConversationButton.setOnClickListener {
            val summary = selectedConversation() ?: return@setOnClickListener showNoConversationToast()
            promptRenameConversation(summary)
        }

        deleteConversationButton.setOnClickListener {
            val summary = selectedConversation() ?: return@setOnClickListener showNoConversationToast()
            AlertDialog.Builder(this)
                .setTitle("Konversation löschen")
                .setMessage("Persistente Konversation \"${summary.title}\" löschen?")
                .setPositiveButton("Löschen") { _, _ ->
                    if (openAIHelper.deletePersistentConversation(summary.id)) {
                        selectedConversationId = null
                        Toast.makeText(this, "Konversation gelöscht", Toast.LENGTH_SHORT).show()
                        refreshConversationList()
                    }
                }
                .setNegativeButton("Abbrechen", null)
                .show()
        }

        refreshConversationsButton.setOnClickListener {
            refreshConversationList()
        }
    }

    private fun refreshConversationList() {
        val summaries = openAIHelper.listPersistentConversations()
        conversationSummaries.clear()
        conversationSummaries.addAll(summaries)

        val active = summaries.firstOrNull { it.isActive }
        selectedConversationId = when {
            active != null -> active.id
            summaries.any { it.id == selectedConversationId } -> selectedConversationId
            else -> null
        }

        conversationDisplayList.clear()
        conversationDisplayList.addAll(summaries.map { formatConversationSummary(it) })
        if (conversationDisplayList.isEmpty()) {
            conversationDisplayList.add("Keine persistenten Konversationen")
        }
        conversationListAdapter.notifyDataSetChanged()

        conversationModeTextView.text = active?.let { "Persistenter Chat: ${it.title}" } ?: "Hauptkonversation: ephemer"
        conversationInfoTextView.text = if (active == null) {
            "Persistente Konversationen werden nur genutzt, wenn du eine erstellst oder auswählst."
        } else {
            "Neue Assistant-Antworten werden in diesem persistenten Chat gespeichert."
        }

        leaveConversationButton.isEnabled = active != null
        renameConversationButton.isEnabled = summaries.isNotEmpty()
        deleteConversationButton.isEnabled = summaries.isNotEmpty()
    }

    private fun formatConversationSummary(summary: PersistentConversationSummary): String {
        val activePrefix = if (summary.isActive) "* " else ""
        val updated = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT, Locale.GERMANY)
            .format(Date(summary.updatedAt))
        return "$activePrefix${summary.title} · ${summary.messageCount} Nachrichten · $updated"
    }

    private fun selectedConversation(): PersistentConversationSummary? {
        val selectedId = selectedConversationId
        return conversationSummaries.firstOrNull { it.id == selectedId }
            ?: conversationSummaries.firstOrNull { it.isActive }
    }

    private fun promptRenameConversation(summary: PersistentConversationSummary) {
        promptConversationTitle(
            title = "Konversation umbenennen",
            prefill = summary.title,
            allowBlank = false,
            positiveLabel = "Speichern"
        ) { title ->
            if (openAIHelper.renamePersistentConversation(summary.id, title)) {
                selectedConversationId = summary.id
                Toast.makeText(this, "Konversation umbenannt", Toast.LENGTH_SHORT).show()
                refreshConversationList()
            }
        }
    }

    private fun promptConversationTitle(
        title: String,
        prefill: String,
        allowBlank: Boolean,
        positiveLabel: String,
        onTitle: (String) -> Unit
    ) {
        val input = EditText(this).apply {
            hint = "Titel"
            setSingleLine(true)
            setText(prefill)
            selectAll()
        }
        AlertDialog.Builder(this)
            .setTitle(title)
            .setView(input)
            .setPositiveButton(positiveLabel) { _, _ ->
                val cleanTitle = input.text.toString().trim()
                if (!allowBlank && cleanTitle.isBlank()) {
                    Toast.makeText(this, "Titel darf nicht leer sein", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                onTitle(cleanTitle)
            }
            .setNegativeButton("Abbrechen", null)
            .show()
    }

    private fun showNoConversationToast() {
        Toast.makeText(this, "Keine persistente Konversation ausgewählt", Toast.LENGTH_SHORT).show()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_CODE_PERMISSIONS -> {
                val bluetoothGranted = AppPermissions.hasBluetoothPermissions(this)
                if (bluetoothGranted) {
                    Log.d(appTag, "Required Bluetooth permissions granted")
                    bluetoothHelper.permissionResult.postValue(true)
                    RokidConnectionService.start(this)
                    if (!AppPermissions.hasAllRuntimePermissions(this)) {
                        Toast.makeText(this, R.string.toast_some_permissions_denied, Toast.LENGTH_LONG).show()
                    }
                } else {
                    Log.d(appTag, "Required Bluetooth permissions denied")
                    bluetoothHelper.permissionResult.postValue(false)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (isWaitingForRokidAuthorization) {
            statusTextView.setText(R.string.status_waiting_rokid_authorization)
        }
    }

    @Deprecated("Used for the Rokid host app authorization contract.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_ROKID_AUTHORIZATION) {
            handleRokidAuthorizationResult(resultCode, data)
        }
    }

    private fun updateDeviceList() {
        deviceList.clear()
        deviceMap.clear()

        // Add bonded devices
        bluetoothHelper.bondedDeviceMap.forEach { (name, device) ->
            val displayName = getString(R.string.device_bonded, name)
            deviceList.add(displayName)
            deviceMap[displayName] = device
        }

        // Add scanned devices
        bluetoothHelper.scanResultMap.forEach { (name, device) ->
            if (!deviceMap.containsValue(device)) {
                deviceList.add(name)
                deviceMap[name] = device
            }
        }

        deviceListAdapter.notifyDataSetChanged()
        statusTextView.text = getString(R.string.status_devices_found, deviceList.size)
    }

    private fun showConnectDialog(device: BluetoothDevice) {
        if (!hasBluetoothPermissions()) {
            Toast.makeText(this, R.string.toast_bluetooth_permission_not_granted, Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val deviceName = device.name ?: "Unknown Device"
            AlertDialog.Builder(this)
                .setTitle(R.string.dialog_authorize_title)
                .setMessage(getString(R.string.dialog_authorize_message, deviceName))
                .setPositiveButton(R.string.dialog_authorize_positive) { _, _ ->
                    requestRokidAuthorization(device)
                }
                .setNegativeButton(R.string.dialog_connect_negative, null)
                .show()
        } catch (e: SecurityException) {
            Log.e(appTag, "SecurityException when accessing device name", e)
            Toast.makeText(this, R.string.toast_bluetooth_permission_not_granted, Toast.LENGTH_SHORT).show()
        }
    }

    private fun requestRokidAuthorization(device: BluetoothDevice) {
        try {
            val deviceName = device.name ?: getString(R.string.device_unknown)
            val hostApp = findInstalledRokidHostApp()

            if (hostApp == null) {
                statusTextView.setText(R.string.status_rokid_host_missing)
                Toast.makeText(this, R.string.toast_rokid_app_not_found, Toast.LENGTH_LONG).show()
                launchRokidStoreSearch()
                return
            }

            pendingAuthorizationDeviceName = deviceName
            isWaitingForRokidAuthorization = true

            statusTextView.text = getString(R.string.status_authorizing_in_rokid, deviceName)
            Toast.makeText(this, R.string.toast_authorize_in_rokid_first, Toast.LENGTH_LONG).show()

            openRokidAuthorizationActivity(hostApp)
        } catch (e: SecurityException) {
            Log.e(appTag, "SecurityException when preparing Rokid authorization", e)
            Toast.makeText(this, R.string.toast_bluetooth_permission_not_granted, Toast.LENGTH_SHORT).show()
        }
    }

    private fun findInstalledRokidHostApp(): RokidHostApp? {
        return rokidAuthorizationHostApps.firstOrNull { hostApp ->
            runCatching {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    packageManager.getPackageInfo(
                        hostApp.packageName,
                        PackageManager.PackageInfoFlags.of(0)
                    )
                } else {
                    @Suppress("DEPRECATION")
                    packageManager.getPackageInfo(hostApp.packageName, 0)
                }
            }.isSuccess
        }
    }

    private fun openRokidAuthorizationActivity(hostApp: RokidHostApp) {
        runCatching {
            val componentIntent = Intent()
                .setComponent(ComponentName(hostApp.packageName, ROKID_AUTH_ACTIVITY_CLASS))
            startActivityForResult(componentIntent, REQUEST_CODE_ROKID_AUTHORIZATION)
        }.recoverCatching {
            val fallbackIntent = Intent(ROKID_AUTH_ACTION).setPackage(hostApp.packageName)
            startActivityForResult(fallbackIntent, REQUEST_CODE_ROKID_AUTHORIZATION)
        }.onSuccess {
            Log.d(appTag, "Opened ${hostApp.displayName} authorization")
        }.onFailure { error ->
            isWaitingForRokidAuthorization = false
            Log.e(appTag, "Could not open ${hostApp.displayName} authorization", error)
            statusTextView.text = getString(
                R.string.status_rokid_authorization_open_failed,
                hostApp.displayName
            )
            Toast.makeText(this, R.string.toast_rokid_authorization_open_failed, Toast.LENGTH_LONG).show()
        }
    }

    private fun launchRokidStoreSearch() {
        val marketIntent = Intent(Intent.ACTION_VIEW, Uri.parse("market://search?q=Hi%20Rokid"))
        val webIntent = Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/search?q=Hi%20Rokid&c=apps"))
        try {
            startActivity(marketIntent)
        } catch (marketError: Exception) {
            try {
                startActivity(webIntent)
            } catch (webError: Exception) {
                Log.e(appTag, "Could not open Hi Rokid or app store", webError)
                Toast.makeText(this, R.string.toast_rokid_app_not_found, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun connectSavedRokidHostSession() {
        val token = RokidHostConnection.savedAuthToken(this) ?: return
        RokidConnectionService.start(this)
        statusTextView.setText(R.string.status_hi_rokid_channel_connecting)
        if (!RokidHostConnection.connect(this, token)) {
            statusTextView.setText(R.string.status_hi_rokid_channel_failed)
        }
    }

    private fun tryConnectAfterRokidAuthorization(authToken: String) {
        isWaitingForRokidAuthorization = false
        RokidHostConnection.saveAuthToken(this, authToken)
        RokidConnectionService.start(this)
        statusTextView.text = getString(
            R.string.status_authorization_return_connecting,
            pendingAuthorizationDeviceName ?: getString(R.string.device_unknown)
        )
        if (!RokidHostConnection.connect(this, authToken)) {
            statusTextView.setText(R.string.status_hi_rokid_channel_failed)
            Toast.makeText(this, R.string.toast_hi_rokid_channel_failed, Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleRokidAuthorizationResult(resultCode: Int, data: Intent?) {
        val authResult = extractRokidAuthResult(data)
        val authToken = extractRokidAuthToken(data)
        if (!authToken.isNullOrBlank() && (authResult == null || authResult == ROKID_AUTH_RESULT_SUCCESS)) {
            Log.d(appTag, "Rokid authorization token received")
            tryConnectAfterRokidAuthorization(authToken)
            return
        }

        isWaitingForRokidAuthorization = false
        Log.w(appTag, "Rokid authorization did not succeed. resultCode=$resultCode, extras=${data?.extras?.keySet()}")
        if (resultCode == Activity.RESULT_CANCELED || authResult == ROKID_AUTH_RESULT_CANCEL) {
            statusTextView.setText(R.string.status_rokid_authorization_cancelled)
        } else {
            statusTextView.setText(R.string.status_rokid_authorization_failed)
        }
    }

    @Suppress("DEPRECATION")
    private fun extractRokidAuthToken(data: Intent?): String? {
        data ?: return null
        data.getStringExtra(ROKID_AUTH_TOKEN_EXTRA)?.takeIf { it.isNotBlank() }?.let { return it }

        val extras = data.extras ?: return null
        extras.keySet().forEach { key ->
            if (key.contains("token", ignoreCase = true)) {
                extras.get(key)?.toString()?.takeIf { it.isNotBlank() }?.let { return it }
            }
        }
        return null
    }

    @Suppress("DEPRECATION")
    private fun extractRokidAuthResult(data: Intent?): Int? {
        val extras = data?.extras ?: return null
        if (!extras.containsKey(ROKID_AUTH_RESULT_EXTRA)) return null
        return when (val value = extras.get(ROKID_AUTH_RESULT_EXTRA)) {
            is Int -> value
            is Number -> value.toInt()
            is String -> value.toIntOrNull()
            else -> null
        }
    }

    private fun updateRokidHostConnectionStatus(cxrConnected: Boolean, glassesConnected: Boolean) {
        val wasConnected = isGlassesConnected
        isGlassesConnected = cxrConnected && glassesConnected
        when {
            isGlassesConnected -> {
                statusTextView.setText(R.string.status_hi_rokid_channel_connected)
                if (!wasConnected) {
                    Toast.makeText(this, R.string.toast_hi_rokid_channel_connected, Toast.LENGTH_SHORT).show()
                }
            }
            cxrConnected -> statusTextView.setText(R.string.status_hi_rokid_channel_waiting_glasses)
            !isWaitingForRokidAuthorization -> statusTextView.setText(R.string.status_hi_rokid_channel_disconnected)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        RokidHostConnection.setConnectionListener(null)
        // Only disconnect the host channel if the activity is finishing (not just being recreated)
        if (isFinishing) {
            bluetoothHelper.release()
            Log.d(appTag, "Activity finishing - Hi Rokid foreground service keeps the channel alive")

            // Clean up temporary audio files
            cleanupTempAudioFiles()
        } else {
            Log.d(appTag, "Activity being recreated - keeping Hi Rokid channel")
        }
    }

    /**
     * Clean up temporary audio files in audio_recordings and tts_audio directories
     */
    private fun cleanupTempAudioFiles() {
        try {
            // Clean up audio recordings directory
            val audioRecordingsDir = getExternalFilesDir("audio_recordings")
            if (audioRecordingsDir != null && audioRecordingsDir.exists()) {
                val deletedCount = deleteDirectoryContents(audioRecordingsDir)
                Log.d(appTag, "Cleaned up audio_recordings: deleted $deletedCount files")
            }

            // Clean up TTS audio directory
            val ttsAudioDir = getExternalFilesDir("tts_audio")
            if (ttsAudioDir != null && ttsAudioDir.exists()) {
                val deletedCount = deleteDirectoryContents(ttsAudioDir)
                Log.d(appTag, "Cleaned up tts_audio: deleted $deletedCount files")
            }
        } catch (e: Exception) {
            Log.e(appTag, "Error cleaning up temporary audio files: ${e.message}", e)
        }
    }

    /**
     * Delete all files in a directory (but keep the directory itself)
     * @param directory The directory to clean
     * @return Number of files deleted
     */
    private fun deleteDirectoryContents(directory: java.io.File): Int {
        var deletedCount = 0
        try {
            directory.listFiles()?.forEach { file ->
                if (file.isFile) {
                    if (file.delete()) {
                        deletedCount++
                        Log.d(appTag, "Deleted file: ${file.name}")
                    } else {
                        Log.w(appTag, "Failed to delete file: ${file.name}")
                    }
                } else if (file.isDirectory) {
                    // Recursively delete subdirectories
                    deletedCount += deleteDirectoryContents(file)
                    if (file.delete()) {
                        Log.d(appTag, "Deleted directory: ${file.name}")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(appTag, "Error deleting directory contents: ${e.message}", e)
        }
        return deletedCount
    }
}
