
// params0 = 0;
// params1 = 0;
// params2 = 0;
// params3 = 0;
// meshsize = 0.002;

Do = 72e-3;                                                            
Di = 51e-3;                                                   
hi = 13e-3;                                                
bli = 3e-3;                                                             
Dc = 3.27640e-2;                                                   
hc = 7.55176e-3;                                                           
ri = 20e-3;                                                    
ra = 18e-3;                                                           
blc = hi-hc; 
rm = ((Dc+params1)*(Dc+params1)+hc*hc-(ri+params0)*(ri+params0))/((Dc+params1)*Sqrt(2)+hc*Sqrt(2)-2*(ri+params0));        
R = rm-ri;
O = rm/Sqrt(2);

Point(0) = {0,0,0,meshsize};
Point(1) = {Dc+params1,0,0,meshsize};
Point(2) = {Di+params2,0,0,meshsize/5};
Point(3) = {Do,0,0,meshsize};
Point(4) = {(ri+params0)/Sqrt(2),(params0+ri)/Sqrt(2),0,meshsize/5};
Point(5) = {Dc+params1,hc,0,meshsize/5};
Point(6) = {Di+params2,hi-bli,0,meshsize/5};
Point(7) = {Do,Do*Tan(Pi/8),0,meshsize};
Point(8) = {Dc+blc,hi+params3,0,meshsize/5};
Point(9) = {Di-bli,hi+params3,0,meshsize/5};
Point(10) = {O,O,0,meshsize};
Point(11) = {Do/Sqrt(2),Do/Sqrt(2),0,meshsize};

//+
Line(1) = {0, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 7};
//+
Line(5) = {11, 7};
//+
Line(6) = {10, 4};
//+
Line(7) = {0, 4};
//+
Line(8) = {1, 5};
//+
Line(9) = {8, 5};
//+
Line(10) = {8, 9};
//+
Line(11) = {9, 6};
//+
Line(12) = {2, 6};
//+
Line(13) = {10, 11};
//+
Ellipse(14) = {4, 10, 10, 5};
//+
Curve Loop(1) = {14, -8, -1, 7};
//+
Surface(1) = {1};
//+
Curve Loop(2) = {13, 5, -4, -3, 12, -11, -10, 9, -14, -6};
//+
Plane Surface(2) = {2};
//+
Physical Curve("GammaN", 15) = {1, 2, 3};
//+
Physical Curve("GammaD", 16) = {6, 13, 5, 4, 7};
//+
Physical Surface("Iron", 17) = {2};
//+
Physical Surface("Air", 18) = {1};
//+
Curve Loop(3) = {10, 11, -12, -2, 8, -9};
//+
Plane Surface(3) = {3};
//+
Physical Surface("Cu", 19) = {3};
