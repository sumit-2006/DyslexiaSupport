package com.example.dysgraphia;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Base64;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.ByteArrayOutputStream;


public class MainActivity extends AppCompatActivity {
    private DrawingView drawingView;
    private Button predictButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // Reference the custom view and button from the layout
        drawingView = findViewById(R.id.drawing_view);
        predictButton = findViewById(R.id.btn_predict);

        predictButton.setOnClickListener(view -> {
            // Get the drawn image from the custom view
            Bitmap drawnImage = drawingView.getBitmap();
            // Convert the Bitmap to a Base64 string to send over HTTP (if using REST API)
            String imageBase64 = convertBitmapToBase64(drawnImage);
            // Start asynchronous task to predict the text using the ML model
            new PredictTask().execute(imageBase64);
        });
    }

    // Convert Bitmap to Base64 String
    private String convertBitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
        byte[] byteArray = outputStream.toByteArray();
        return Base64.encodeToString(byteArray, Base64.DEFAULT);
    }

    // AsyncTask to simulate a network call to your Python-based ML model
    private class PredictTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            String imageData = params[0];
            // Simulate calling your Python ML model via a REST API or embedded Python code
            return MLModel.predictFromImageData(imageData);
        }

        @Override
        protected void onPostExecute(String predictionResult) {
            Toast.makeText(MainActivity.this, "Prediction: " + predictionResult, Toast.LENGTH_LONG).show();
        }
    }
}