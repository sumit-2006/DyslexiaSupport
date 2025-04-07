package com.example.dysgraphia;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawingView extends View {
    private Path path = new Path();
    private Paint paint = new Paint();
    private float lastX, lastY;

    // Constructor for creating view in code
    public DrawingView(Context context) {
        super(context);
        init();
    }

    // Constructor for inflating from XML
    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint.setAntiAlias(true);
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5f);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(path, paint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        int toolType = event.getToolType(0); // Detect input tool type
        if (toolType == MotionEvent.TOOL_TYPE_STYLUS || toolType == MotionEvent.TOOL_TYPE_FINGER) {
            float x = event.getX();
            float y = event.getY();
            float pressure = event.getPressure(); // Pressure sensitivity

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    path.moveTo(x, y);
                    lastX = x;
                    lastY = y;
                    return true;
                case MotionEvent.ACTION_MOVE:
                    // Optionally adjust stroke width based on pressure:
                    paint.setStrokeWidth(5f + (pressure * 10));
                    path.lineTo(x, y);
                    break;
                case MotionEvent.ACTION_UP:
                    // Optionally finalize the path segment here.
                    break;
            }
            invalidate();
        }
        return true;
    }

    // Method to capture the drawing as a Bitmap image
    public Bitmap getBitmap() {
        Bitmap bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        this.draw(canvas);
        return bitmap;
    }
}
