package com.example.cloudvisionis31082023;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.media.ThumbnailUtils;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import android.Manifest;
import android.content.pm.PackageManager;

import com.example.cloudvisionis31082023.ml.ModelUnquant;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity
        implements OnSuccessListener<Text>, OnFailureListener {
    public static int REQUEST_CAMERA = 111;
    public static int REQUEST_GALLERY = 222;
    public int tamanio_imagen = 224;
    ArrayList<String> permisosNoAprobados;
    TextView txtResults;
    ImageView mImageView;
    Bitmap mSelectedImage;
    Button btnCamara, btnGaleria;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        txtResults = findViewById(R.id.txtresults);
        mImageView = findViewById(R.id.image_view);

        btnCamara = findViewById(R.id.btCamera);
        btnGaleria = findViewById(R.id.btGallery);

        ArrayList<String> permisos_requeridos = new ArrayList<String>();
        permisos_requeridos.add(Manifest.permission.CAMERA);
        permisos_requeridos.add(Manifest.permission.MANAGE_EXTERNAL_STORAGE);
        permisos_requeridos.add(Manifest.permission.READ_EXTERNAL_STORAGE);

        permisosNoAprobados  = getPermisosNoAprobados(permisos_requeridos);
        requestPermissions(permisosNoAprobados.toArray(new String[permisosNoAprobados.size()]),
                100);
    }

    public void abrirGaleria (View view){
        Intent i = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }

    public void abrirCamera (View view){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && null != data) {
            try {
                if (requestCode == REQUEST_CAMERA)
                    mSelectedImage = (Bitmap) data.getExtras().get("data");
                else
                    mSelectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());

                mImageView.setImageBitmap(mSelectedImage);

                //Llamada a función para reconocer el rostro apenas cargue la imagen seleccionada.
                Bitmap imagen = Bitmap.createScaledBitmap(mSelectedImage, tamanio_imagen, tamanio_imagen, false);
                Reconocer(imagen);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        for(int i=0; i<permissions.length; i++){
            if(permissions[i].equals(Manifest.permission.CAMERA)){
                //btnCamara.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            } else if(permissions[i].equals(Manifest.permission.MANAGE_EXTERNAL_STORAGE) ||
                    permissions[i].equals(Manifest.permission.READ_EXTERNAL_STORAGE)
            ) {
                //btnGaleria.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            }
        }
    }

    public void btn_Reconocer(View view){
        //Instanciar el modelo basado en el creado (tensor flow lite)
        BitmapDrawable imagenBase = (BitmapDrawable) mImageView.getDrawable();

        //Definir un bitmap para enviarlo al modelo creado.
        Bitmap image = imagenBase.getBitmap().copy(imagenBase.getBitmap().getConfig(), false);
        image = Bitmap.createScaledBitmap(image, tamanio_imagen, tamanio_imagen, true);

        Reconocer(image);
    }

    public void Reconocer(Bitmap imagen){
        try {
            //Instanciar el modelo basado en el creado (tensor flow lite)
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());
            BitmapDrawable imagenBase = (BitmapDrawable) mImageView.getDrawable();

            //Definir un bitmap para enviarlo al modelo creado.
            imagen = Bitmap.createScaledBitmap(imagen, tamanio_imagen, tamanio_imagen, true);

            //Establecer las dimensiones que tendrá la imágen.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, tamanio_imagen, tamanio_imagen, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * tamanio_imagen * tamanio_imagen * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            //Obtener los pixeles de la imagen e iterar cada uno de los pixeles para obtener el color respectivo
            //en formato RGB (Red, green, blue).
            int[] intValues = new int[tamanio_imagen * tamanio_imagen];
            imagen.getPixels(intValues, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());

            int pixel = 0;

            for(int i = 0; i <  imagen.getHeight(); i ++){
                for(int j = 0; j < imagen.getWidth(); j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            //Cargar los bytes obtenidos al modelo de tensorflow.
            inputFeature0.loadBuffer(byteBuffer);

            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //Obtener los resultados correspondientes a las neuronas existentes en el modelo.
            float[] confidences = outputFeature0.getFloatArray();

            //Definir las clases o neuronas del modelo.
            String[] classes = {"Gerard Way", "Brendon Urie", "Freddie Mercury"};

            //Unificar el nombre de la clase y el porcentaje correspondiente.
            String resultados = "";
            DecimalFormat df = new DecimalFormat("0.00");
            for(int i = 0; i < classes.length; i++){
                resultados += classes[i] + ", " + (df.format(confidences[i] * 100)) + "%" + "\n";
            }

            txtResults.setText("Resultados" + "\n" + resultados);


            //Finalmente, cerrar el modelo.
            model.close();
        } catch (Exception e) {
            txtResults.setText(e.getMessage());
        }
    }

    public ArrayList<String> getPermisosNoAprobados(ArrayList<String>  listaPermisos) {
        ArrayList<String> list = new ArrayList<String>();

        if (Build.VERSION.SDK_INT >= 23)
            for(String permiso: listaPermisos) {
                if (checkSelfPermission(permiso) != PackageManager.PERMISSION_GRANTED) {
                    list.add(permiso);
                }
            }


        return list;
    }

    public void OCRfx(View v) {
        InputImage image = InputImage.fromBitmap(mSelectedImage, 0);
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        recognizer.process(image)
                .addOnSuccessListener(this)
                .addOnFailureListener(this);
    }

    @Override
    public void onFailure(@NonNull Exception e) {
        txtResults.setText("Error al procesar imagen");
    }

    @Override
    public void onSuccess(Text text) {
        List<Text.TextBlock> blocks = text.getTextBlocks();
        String resultados="";
        if (blocks.size() == 0) {
            resultados = "No hay Texto";
        }else{
            for (int i = 0; i < blocks.size(); i++) {
                List<Text.Line> lines = blocks.get(i).getLines();
                for (int j = 0; j < lines.size(); j++) {
                    List<Text.Element> elements = lines.get(j).getElements();
                    for (int k = 0; k < elements.size(); k++) {
                        resultados = resultados + elements.get(k).getText() + " ";
                    }
                }
            }
            resultados=resultados + "\n";
        }
        txtResults.setText(resultados);

    }


    public void Rostrosfx(View v) {
        InputImage image = InputImage.fromBitmap(mSelectedImage, 0);
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        .build();

        FaceDetector detector = FaceDetection.getClient(options);
        detector.process(image)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        if (faces.size() == 0) {
                            txtResults.setText("No Hay rostros");
                        }else{
                            txtResults.setText("Hay " + faces.size() + " rostro(s)");
                        }

                        BitmapDrawable drawable = (BitmapDrawable)mImageView.getDrawable();
                        Bitmap bitmap = drawable.getBitmap().copy(Bitmap.Config.ARGB_8888, true);
                        Canvas canvas = new Canvas(bitmap);
                        Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setTextSize(70);
                        paint.setStrokeWidth(20);
                        paint.setStyle(Paint.Style.STROKE);

                        for(Face face: faces){
                            canvas.drawRect(face.getBoundingBox(), paint);
                            /*for (FaceContour contorno: face.getAllContours()){
                                for (PointF punto: contorno.getPoints()){
                                    //punto.x, punto.y
                                }
                            }*/
                        }

                        mImageView.setImageBitmap(bitmap);
                    }
                })
                .addOnFailureListener(this);
    }


}