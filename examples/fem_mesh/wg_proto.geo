
w1 = 2.0;
w2 = 3.0;
h = 0.5;
ry = 1.5;
rz = 1.5;


Point(1) = {w1*0.5,0,-3,meshsize};
Point(2) = {w1*0.5,0,-2+theta1,meshsize};
Point(3) = {w2*0.5,0,-1+theta2,meshsize};
Point(4) = {w2*0.5,0,0,meshsize};
Point(5) = {w2*0.5,ry,rz+theta3,meshsize};
Point(6) = {w2*0.5,ry,0,meshsize};

Point(7) =  {w1*0.5,h,-3,meshsize};
Point(8) =  {w1*0.5,h,-2+theta1,meshsize};
Point(9) =  {w2*0.5,h,-1+theta2,meshsize};
Point(10) = {w2*0.5,h,0,meshsize};
Point(11) = {w2*0.5,ry,rz+theta3-h,meshsize};
Point(12) = {w2*0.5,ry,0,meshsize};

Point(13) = {-w1*0.5,0,-3,meshsize};
Point(14) = {-w1*0.5,0,-2+theta1,meshsize};
Point(15) = {-w2*0.5,0,-1+theta2,meshsize};
Point(16) = {-w2*0.5,0,0,meshsize};
Point(17) = {-w2*0.5,ry,rz+theta3,meshsize};
Point(18) = {-w2*0.5,ry,0,meshsize};

Point(19) =  {-w1*0.5,h,-3,meshsize};
Point(20) =  {-w1*0.5,h,-2+theta1,meshsize};
Point(21) =  {-w2*0.5,h,-1+theta2,meshsize};
Point(22) = {-w2*0.5,h,0,meshsize};
Point(23) = {-w2*0.5,ry,rz+theta3-h,meshsize};
Point(24) = {-w2*0.5,ry,0,meshsize};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};

Line(4) = {7,8};
Line(5) = {8,9};
Line(6) = {9,10};

Line(7) = {13,14};
Line(8) = {14,15};
Line(9) = {15,16};

Line(10) = {19,20};
Line(11) = {20,21};
Line(12) = {21,22};


//+
Line(13) = {7, 1};
//+
Line(14) = {19, 13};
//+
Line(15) = {13, 1};
//+
Line(16) = {8, 20};
//+
Line(17) = {2, 14};
//+
Line(18) = {8, 2};
//+
Line(19) = {20, 14};
//+
Line(20) = {21, 15};
//+
Line(21) = {15, 3};
//+
Line(22) = {9, 3};
//+
Line(23) = {9, 21};
//+
Line(24) = {10, 22};
//+
Line(25) = {16, 4};
//+
Line(26) = {10, 4};
//+
Line(27) = {22, 16};
//+
Line(28) = {5, 11};
//+
Line(29) = {23, 17};
//+
Line(30) = {17, 5};
//+
Line(31) = {11, 23};
//+
Ellipse(32) = {10, 6, 6, 11};
//+
Ellipse(33) = {22, 18, 18, 23};
//+
Ellipse(34) = {4, 6, 6, 5};
//+
Ellipse(35) = {16, 18, 18, 17};
//+
Line(36) = {7, 19};
//+
Curve Loop(1) = {10, -16, -4, 36};
//+
Surface(1) = {1};
//+
Curve Loop(2) = {1, 17, -7, 15};
//+
Surface(2) = {2};
//+
Curve Loop(3) = {18, 17, -19, -16};
//+
Surface(3) = {3};
//+
Curve Loop(4) = {36, 14, 15, -13};
//+
Surface(4) = {4};
//+
Curve Loop(5) = {5, 22, -2, -18};
//+
Surface(5) = {5};
//+
Curve Loop(6) = {21, -2, 17, 8};
//+
Surface(6) = {6};
//+
Curve Loop(7) = {23, 20, 21, -22};
//+
Surface(7) = {7};
//+
Curve Loop(8) = {11, 20, -8, -19};
//+
Surface(8) = {8};
//+
Curve Loop(9) = {23, -11, -16, 5};
//+
Surface(9) = {9};
//+
Curve Loop(10) = {6, 24, -12, -23};
//+
Surface(10) = {10};
//+
Curve Loop(11) = {25, -3, -21, 9};
//+
Surface(11) = {11};
//+
Curve Loop(12) = {24, 27, 25, -26};
//+
Surface(12) = {12};
//+
Curve Loop(13) = {12, 27, -9, -20};
//+
Surface(13) = {13};
//+
Curve Loop(14) = {31, 29, 30, 28};
//+
Surface(14) = {14};
//+
Curve Loop(15) = {32, 31, -33, -24};
//+
Surface(15) = {15};
//+
Curve Loop(16) = {34, -30, -35, 25};
//+
Surface(16) = {16};
//+
Curve Loop(17) = {34, 28, -32, 26};
//+
Surface(17) = {17};
//+
Curve Loop(18) = {33, 29, -35, -27};
//+
Surface(18) = {18};
//+
Physical Surface("input",101) = {4};
//+
Physical Surface("output",102) = {14};
//+
Curve Loop(19) = {3, -26, -6, 22};
//+
Plane Surface(19) = {19};
//+
Curve Loop(20) = {4, 18, -1, -13};
//+
Plane Surface(20) = {20};
//+
Curve Loop(21) = {10, 19, -7, -14};
//+
Plane Surface(21) = {21};
//+
Surface Loop(1) = {15, 17, 16, 14, 18, 12};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {10, 19, 11, 13, 7, 12};
//+
Volume(2) = {2};
//+
Surface Loop(3) = {9, 8, 6, 5, 7, 3};
//+
Volume(3) = {3};
//+
Surface Loop(4) = {1, 21, 2, 20, 4, 3};
//+
Volume(4) = {4};
//+
Physical Surface("pec",104) = {17, 16, 15, 18, 13, 10, 11, 19, 5, 9, 6, 8, 21, 1, 2, 20};

//+
Physical Volume("material", 105) = {3};
//+
Physical Volume("material0", 106) = {2, 1, 4};
