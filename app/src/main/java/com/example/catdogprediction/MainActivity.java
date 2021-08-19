package com.example.catdogprediction;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.catdogprediction.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private ImageView predictionImageView;
    private Button uploadButton;
    private Button predictButton;
    private TextView predictionResultTextview;
    private TextView frameText;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        predictionImageView = findViewById(R.id.image);
        uploadButton = findViewById(R.id.uploadBtn);
        predictButton = findViewById(R.id.predictBtn);
        predictionResultTextview = findViewById(R.id.resultTextView);
        frameText = findViewById(R.id.frameText);

        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 200);
            }
        });

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (predictionImageView.getDrawable() == null) {
                    Toast.makeText(MainActivity.this, "Please Upload an Image First", Toast.LENGTH_SHORT).show();
                } else {
                    img = Bitmap.createScaledBitmap(img, 128, 128, true);

                    try {
                        Model model = Model.newInstance(getApplicationContext());

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);

                        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                        tensorImage.load(img);
                        ByteBuffer byteBuffer = tensorImage.getBuffer();

                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        Model.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        // Releases model resources if no longer used.
                        model.close();

//                    predictionResultTextview.setText(outputFeature0.getFloatArray()[0] + "\n" + outputFeature0.getFloatArray()[1]);
                        if (outputFeature0.getFloatArray()[0] == 1.0 && outputFeature0.getFloatArray()[1] == 0.0) {
                            predictionResultTextview.setText("Cat");
                        } else if (outputFeature0.getFloatArray()[0] == 0.0 && outputFeature0.getFloatArray()[1] == 1.0) {
                            predictionResultTextview.setText("Dog");
                        } else if (outputFeature0.getFloatArray()[0] == 0.0 && outputFeature0.getFloatArray()[1] < 1.0 && outputFeature0.getFloatArray()[1] > 0.0) {
                            predictionResultTextview.setText("Dog");
                        } else if (outputFeature0.getFloatArray()[1] == 0.0 && outputFeature0.getFloatArray()[0] < 1.0 && outputFeature0.getFloatArray()[0] > 0.0) {
                            predictionResultTextview.setText("Cat");
                        } else {
                            predictionResultTextview.setText("Neither a Cat or Dog");
                        }

                    } catch (IOException e) {
                        // TODO Handle the exception
                    }

                }
            }
        });
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==200){
            predictionImageView.setImageURI(data.getData());
            frameText.setVisibility(View.GONE);

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }
}