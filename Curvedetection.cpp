
// Read the image as gray-scale
Mat img = imread('lanes.jpg', IMREAD_COLOR);
// Convert to gray-scale
Mat gray = cvtColor(img, COLOR_BGR2GRAY);
// Store the edges
Mat edges;
// Find the edges in the image using canny detector
Canny(gray, edges, 50, 200);
// Create a vector to store lines of the image
vector<Vec4i> lines;
// Apply Hough Transform
HoughLinesP(edges, lines, 1, CV_PI/180, thresh, 10, 250);
// Draw lines on the image
for (size_t i=0; i<lines.size(); i++) {
    Vec4i l = lines[i];
    line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);
}
// Show result image
imshow("Result Image", img);







// Read the image as gray-scale
img = imread("circles.png", IMREAD_COLOR);
// Convert to gray-scale
gray = cvtColor(img, COLOR_BGR2GRAY);
// Blur the image to reduce noise
Mat img_blur;
medianBlur(gray, img_blur, 5);
// Create a vector for detected circles
vector<Vec3f>  circles;
// Apply Hough Transform
HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img.rows/64, 200, 10, 5, 30);
// Draw detected circles
for(size_t i=0; i<circles.size(); i++) {
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    circle(img, center, radius, Scalar(255, 255, 255), 2, 8, 0);
}


