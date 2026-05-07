;
; 
; ----  SPC EFFECTIVE AREA MODELS  -----
; ANALYTIC (cold plasma) and
; NUMERICAL CONVOLUTION (warm plasma)
; 
;
;
; USAGE:
;    areas = psp_spc_swp_coldspot(phi, theta)
;    areas = psp_spc_swp_warmspot(phi, theta, w, n=n)
;
; INPUTS:
;    phi = horizontal incidence angle (in DEGREES)
;    theta = vertical incidence angle (in DEGREES)
;    w = spreading factor, equal to tan( spread angle )
; 
;    n = number of transformation points used in 
;        gaussian quadrature formulation of the warm
;        plasma integral (6, 10, 20, 48, 96).
;   
;        ROUGHLY: n=6 is precise to ~1 mm^2
;                 n=10 is precise to ~0.1 mm^2
;                 n=20 is precise to ~0.01 mm^2
;
;        higher precision integration is NOT warranted
;        because (1) element dimensions are specified to ~0.1%
;        (2) warm plasma spread is truncated ("paired") 
;        at 3*sigma
;
; NOTES:
;
;     The spreading factor w is relatable to a maxwellian
;     most probable thermal speed via w = (v_th / v_z)
;
;     Cup geometry is presently specified in the setparams 
;     function, where the numbers are hard-coded. Thise 
;     procedure should be overwritten with an appropriate 
;     tabe lookup.    
;

pro setparams, rl=rl, re=re, rc=rc, ze=ze, zl=zl, epsilon=epsilon, $
  dz_coll=dz_coll, al=al, ac=ac, amax=amax

re=39.9d
rl=10.86d
rc=23.94d
ze=70.5d
zl=16.9d
epsilon = 0.36d
dz_coll = 1.981d

aL = atan( (re-rl), (ze-zl) ) /!dtor
aC = atan( (rc-rl), zl) / !dtor
amax = atan( (rc+re), ze ) /!dtor

;IDL> print, al/!dtor, ac/!dtor, amax/!dtor
;      28.4484      37.7386      42.1618

end



;------------------------------------------------------------
;------------------------------------------------------------
;
;
;                 GEOMETRY CALCULATIONS            
;
;
;------------------------------------------------------------
;------------------------------------------------------------


; integrate the area under an semicircle in the (x,y) plane with respect to
; x over a finite domain [x0, x1]
function int_uppersemicircle_dx, r=r, xc=xc, yc=yc, x0=x0, x1=x1

  if n_elements(r) eq 0 then r = 1.   ; if no arg, use unit circle
  if n_elements(xc) eq 0 then xc = 0. ; if no offset, assume centered
  if n_elements(yc) eq 0 then yc = 0. ; if not offset, assume centered
  if n_elements(x0) eq 0 then x0 = xc-abs(r) ; if no bound provided, integrate whole range
  if n_elements(x1) eq 0 then x1 = xc+abs(r) ; if no bound provided, integrate whole range
  if x1 eq x0 then return, 0.            ; if range size is zero, return zero

                                ; if bounds are out of order, swap them
  if x1 lt x0 then begin
     lower = x1
     upper = x0
  endif else begin
     lower = x0 
     upper = x1
  endelse

  if lower ge (xc + abs(r)) then return, 0. ; if range is entirely outside of the circle
  if upper le (xc - abs(r)) then return, 0. ; if range is entirely outside of the circle
  if lower lt (xc - abs(r)) then lower = xc - abs(r) ; if bound is beyond circle domain
  if upper gt (xc + abs(r)) then upper = xc + abs(r) ; if bound is beyond circle domai
  num_upper = upper - xc
  num_lower = lower - xc  
                                ; check these args to make sure we don't try to take the 
; square root of a very small (possibly negative) number instead of 
; a zero. I have put in an ad hoc tolerance of 1 part in a
; billion. I don't know if that is sufficient
  denom_uparg =  -xc^2 + r^2 + 2.*xc*upper - upper^2
  denom_loarg =  -xc^2 + r^2 + 2.*xc*lower - lower^2
  if denom_uparg gt (-1e-9) then denom_uparg = denom_uparg > 0.
  if denom_loarg gt (-1e-9) then denom_loarg = denom_loarg > 0.
  if (r^2+ 2.*xc*upper) EQ (upper^2 + XC^2)  then denom_upper = 0. else $
     denom_upper = sqrt(denom_uparg)
  if (r^2+ 2.*xc*lower) EQ (lower^2 + XC^2) then denom_lower = 0. else $
     denom_lower = sqrt(denom_loarg)

  arg_upper = yc*upper + $
              0.5*num_upper*denom_upper + $
              0.5*(r^2)*atan( num_upper, denom_upper)
  arg_lower = yc*lower + $
              0.5*num_lower*denom_lower + $
              0.5*(r^2)*atan( num_lower, denom_lower)
  if (finite(arg_upper - arg_lower, /nan) eq 1) or (arg_upper eq arg_lower) then return, 0.
  return, arg_upper - arg_lower

end

 ; integrate the area under an semicircle in the (x,y) plane with respect to
; x over a finite domain [x0, x1]
function int_lowersemicircle_dx, r=r, xc=xc, yc=yc, x0=x0, x1=x1

  if n_elements(r) eq 0 then r = 1.   ; if no arg, use unit circle
  if n_elements(xc) eq 0 then xc = 0. ; if no offset, assume centered
  if n_elements(yc) eq 0 then yc = 0. ; if not   num_upper = xc-upper
  if n_elements(x0) eq 0 then x0 = xc-abs(r) ; if no bound provided, integrate whole range
  if n_elements(x1) eq 0 then x1 = xc+abs(r) ; if no bound provided, integrate whole range
  if x1 eq x0 then return, 0.            ; if range size is zero, return zero

                                ; if bounds are out of order, swap them
  if x1 lt x0 then begin
     lower = x1
     upper = x0
  endif else begin
     lower = x0 
     upper = x1
  endelse

  if lower ge (xc + abs(r)) then return, 0. ; if range is entirely outside of the circle
  if upper le (xc - abs(r)) then return, 0. ; if range is entirely outside of the circle
  if lower lt (xc - abs(r)) then lower = xc - abs(r) ; if bound is beyond circle domain
  if upper gt (xc + abs(r)) then upper = xc + abs(r) ; if bound is beyond circle domain

; these checks need some kind of finite difference fix. There are 
; zeros that are coming out -1e-10 or so and crashing it. What is a
; good roundoff?
  num_upper = xc-upper
  num_lower = xc-lower
  denom_uparg =  -xc^2 + r^2 + 2.*xc*upper - upper^2
  denom_loarg =  -xc^2 + r^2 + 2.*xc*lower - lower^2
  if denom_uparg gt (-1e-9) then denom_uparg = denom_uparg > 0.
  if denom_loarg gt (-1e-9) then denom_loarg = denom_loarg > 0.

  if (r^2+ 2.*xc*upper) EQ (upper^2 + XC^2)  then denom_upper = 0. else $
     denom_upper = sqrt(denom_uparg)
  if (r^2+ 2.*xc*lower) EQ (lower^2 + XC^2) then denom_lower = 0. else $
     denom_lower = sqrt(denom_loarg)

  arg_upper = yc*upper + $
              0.5*num_upper*denom_upper + $
              0.5*(r^2)*atan( num_upper, denom_upper)
  arg_lower = yc*lower + $
              0.5*num_lower*denom_lower + $
              0.5*(r^2)*atan( num_lower, denom_lower)
  if (finite(arg_upper - arg_lower, /nan) eq 1) or (arg_upper eq arg_lower) then return, 0.
  return, arg_upper - arg_lower

end

  
; find the x-coordinates for the intersection points between two circles
function intersection_of_circles, x1=x1, y1=y1, r1=r1, x2=x2, y2=y2, r2=r2

if n_elements(x1)*n_elements(x2)*n_elements(r1)*n_elements(r2)*$
   n_elements(y1)*n_elements(y2) eq 0 then begin
   print, 'Error: intersection_of_circles requires all six arguments'
   return, [!values.f_nan, !values.f_nan]
endif

; distance between circle centers
d=sqrt( (x2-x1)^2 + (y2-y1)^2 )

; if they are the same circle, they intersect EVERYWHERE.
; not sure the best way to handle this, for now let's return the 
; extremes since we're using this for area of overlap
if (d eq 0.) and (abs(r1) eq abs(r2)) then return, [ x1-r1, x1+r1 ]

; if circles are too far apart, there are no intersections
if d gt (abs(r2)+abs(r1)) then return, [!values.f_nan, !values.f_nan]

; if one circle is entirely within the other, there are no intersections
if d lt abs( abs(r2) - abs(r1)) then return, [!values.f_nan, !values.f_nan]

; distance from circle 1 center to midpoint between the two circles
a= (r1^2 - r2^2 + d^2)/(2.*d)

; distance from intersection to midpoint between the two circles
h= sqrt(r1^2 - a^2)

; location of midpoint between the two circles
x3= x1 + a*(x2-x1)/d   
y3= y1 + a*(y2-y1)/d  

; x at the two intersections
x4= x3 + h*(y2-y1)/d    ;   // also x3=x2-h*(y1-y0)/d
x5= x3 - h*(y2-y1)/d    

return, [x4<x5, x4>x5]

end

; find the are under the curve in the region of 2-circle overlap
function area_of_intersection,x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2

intercepts = intersection_of_circles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
xmax = ( x1 + abs(r1) ) < ( x2 + abs(r2) ) 
xmin =  ( x1 - abs(r1) ) > ( x2 - abs(r2) ) 
xpoints = [xmin, intercepts, xmax]

xi = xpoints[where(finite(xpoints, /nan) eq 0)]

area = 0.
for i = 0, n_elements(xi)-2 do begin
   low1 = int_lowersemicircle_dx(r=r1, xc = x1, yc = y1, x0 = xi[i], x1 = xi[i+1])
   low2 = int_lowersemicircle_dx(r=r2, xc = x2, yc = y2, x0 = xi[i], x1 = xi[i+1])
   high1 = int_uppersemicircle_dx(r=r1, xc = x1, yc = y1, x0 = xi[i], x1 = xi[i+1])
   high2 = int_uppersemicircle_dx(r=r2, xc = x2, yc = y2, x0 = xi[i], x1 = xi[i+1])
   thisarea = (high1 < high2) - (low1 > low2)
   area = area + (thisarea > 0.)
endfor

;stop 
return, area

end

function circle_x_at_yeq, y, xc=xc, yc=yc, r=r

arg = r^2 - (y-yc)^2
if arg lt 0. then return, [!values.f_nan, !values.f_nan] $
else return, [xc - sqrt(r^2 - (y-yc)^2), xc + sqrt(r^2 - (y-yc)^2)]

end



function circle_xintercepts, xc=xc, yc=yc, r=r

  ; function is (y-yc)^2 + (x-xc)^2 = r^2
  ; becomes x = xc +- sqrt(r^2 - yc^2)

arg = r^2 - yc^2
if arg lt 0. then return, [!values.f_nan, !values.f_nan] $
else return, [xc - sqrt(r^2 - yc^2), xc + sqrt(r^2 - yc^2)]

end



; find the are under the curve in the region of 2-circle overlap
function quadrant_area_of_intersection,x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2, x3=x3, y3=y3, r3=r3, $
                                       epsilon=epsilon

case n_elements(epsilon) of 
   0: begin
      eps_x = 0.
      eps_y = 0.
   end
   1: begin
      eps_x = epsilon
      eps_y = epsilon
   end
   2: begin
      eps_x = epsilon[0]
      eps_y = epsilon[1]
   end
   else: begin
      eps_x = 0.
      eps_y = 0.
   endelse
endcase

; if the third circle isn't specified, 
; the perfectly good hack is to make it a copy of the first
if n_elements(r3) eq 0 then begin
   x3=x1
   y3=y1
   r3=r1
endif

; a list of everywhere that the three circles intersect one another
; (1x2, 2x3, 1x3)
intercepts1 = intersection_of_circles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
intercepts2 = intersection_of_circles(x1=x1, x2=x3, y1=y1, y2=y3, r1=r1, r2=r3)
intercepts3 = intersection_of_circles(x1=x2, x2=x3, y1=y2, y2=y3, r1=r2, r2=r3)
intercepts = [intercepts1, intercepts2, intercepts3]

; a list of the X intercepts of all three circles
x_intercepts1 = circle_x_at_yeq(eps_y/2., xc=x1, yc=y1, r=r1)
x_intercepts2 = circle_x_at_yeq(eps_y/2., xc=x2, yc=y2, r=r2)
x_intercepts3 = circle_x_at_yeq(eps_y/2., xc=x3, yc=y3, r=r3)
x_intercepts4 = circle_x_at_yeq(-eps_y/2., xc=x1, yc=y1, r=r1)
x_intercepts5 = circle_x_at_yeq(-eps_y/2., xc=x2, yc=y2, r=r2)
x_intercepts6 = circle_x_at_yeq(-eps_y/2., xc=x3, yc=y3, r=r3)
x_intercepts = [x_intercepts1, x_intercepts2, x_intercepts3, x_intercepts4, x_intercepts5, x_intercepts6]

; the lower and upper bounds of the intersecting area
; defined as the maximum leftmost extent and the minimum rightmost extent
xmax = min([x1 + abs(r1) ,  x2 + abs(r2) , x3 + abs(r3)])
xmin = max([x1 - abs(r1) ,  x2 - abs(r2) , x3 - abs(r3)])
if xmin ge xmax then return, [0.,0.,0.,0.]

; collect all of these points. These define the segments for piecewise
; integration. Between any two points, we can be sure that no two
; circles cross and that no circle crosses the x-axis.
;
; This implies that the integral of the lowest(highest) curve
; will always be the of the lowest(highest) magnitude on any given segment.
xpoints = [xmin, intercepts, xmax, 0, x_intercepts, (-eps_x/2.), (eps_x/2.)]
in_domain = where((finite(xpoints, /nan) eq 0) and (xpoints ge xmin) and (xpoints le xmax), nin)
if nin lt 2 then return, [0.,0.,0.,0.] 

xi = xpoints[where((finite(xpoints, /nan) eq 0) and (xpoints ge xmin) and (xpoints le xmax))]
xi = xi[sort(xi)]

; pick out the -x set and the +x set
rhs = where(xi ge (eps_x/2.), nrhs)
lhs = where(xi le (-eps_x/2.), nlhs)

; initialize areas for our tally
area_pp = 0.
area_mp = 0.
area_pm = 0.
area_mm = 0.
area = 0.

; integrate over the +x half plane
if nrhs ge 2 then begin
   low1 = fltarr(nrhs-1)
   low2 = fltarr(nrhs-1)
   low3 = fltarr(nrhs-1)
   high1 = fltarr(nrhs-1)
   high2 = fltarr(nrhs-1)
   high3 = fltarr(nrhs-1)
   gap = fltarr(nrhs-1)
   xip = xi[rhs]                ; limits for +x integral segments
   for i = 0, nrhs-2 do begin &$
                                ; lower half circle integrals on this range
      low1[i] = int_lowersemicircle_dx(r=r1, xc = x1, yc = y1, $
                                       x0 = xip[i], x1 = xip[i+1]) &$
      low2[i] = int_lowersemicircle_dx(r=r2, xc = x2, yc = y2, $
                                       x0 = xip[i], x1 = xip[i+1]) &$
      low3[i] = int_lowersemicircle_dx(r=r3, xc = x3, yc = y3, $
                                       x0 = xip[i], x1 = xip[i+1]) &$

                                ; upper half circle integrals on this range
      high1[i] = int_uppersemicircle_dx(r=r1, xc = x1, yc = y1, $
                                        x0 = xip[i], x1 = xip[i+1]) &$
      high2[i] = int_uppersemicircle_dx(r=r2, xc = x2, yc = y2, $
                                        x0 = xip[i], x1 = xip[i+1]) &$
      high3[i] = int_uppersemicircle_dx(r=r3, xc = x3, yc = y3, $
                                        x0 = xip[i], x1 = xip[i+1]) &$

                                ; gap integral on this range
      gap[i] = (eps_y/2.)*(xip[i+1] - xip[i])
   endfor

                                ; the total area is the upper half -
                                ; lower half integral, if it is
                                ; positive. If it is not positive,
                                ; that means that the lower curve is
                                ; above the upper curve, i.e. there is
                                ; no overlap region
   low = max([[low1], [low2], [low3]], dim = 2)
   high = min([[high1], [high2], [high3]], dim = 2)
   if total( finite(low, /nan) ) + total( finite(high, /nan)) gt 0. then stop ; consistency check
 
   dAp = ( ((high - gap) > 0.) - ((low - gap) > 0.) ) > 0.                     ; area segments in upper half plane
   dAm = ( ((high + gap) < 0.) - ((low + gap) < 0.) ) > 0.                     ; area segments in lower half plane
   dA = dAp + dAm
 
   area = area + total(dA)
   area_pp = area_pp + total(dAp)
   area_pm = area_pm + total(dAm)

;stop

;      thisarea = min([high1, high2, high3]) - max([low1, low2, low3])
;      area = area + (thisarea > 0.) ; add piecewise integral segment to the sum
;      ; for the upper and lower quadrants, we put a floor/ceiling at zero.
;      thisarea_p = ( min([high1,high2,high3]) > 0. ) - ( max([low1, low2, low3]) > 0. )
;      thisarea_m = ( min([high1,high2,high3]) < 0. ) - ( max([low1, low2, low3]) < 0. )
;      if thisarea ne (thisarea_p + thisarea_m) then stop ; consistency check
;      if (finite(thisarea, /nan) eq 1) or $
;         (finite(thisarea_p, /nan) eq 1) or $
;         (finite(thisarea_m, /nan) eq 1) then stop; consistency check;
;      area_pp = area_pp + (thisarea_p > 0.) ; add piecewise integral segment to the sum
;      area_pm = area_pm + (thisarea_m > 0.) ; add piecewise integral segment to the sum
;   endfor
endif

if nlhs ge 2 then begin

   xim = xi[lhs]
   low1 = fltarr(nlhs-1)
   low2 = fltarr(nlhs-1)
   low3 = fltarr(nlhs-1)
   high1 = fltarr(nlhs-1)
   high2 = fltarr(nlhs-1)
   high3 = fltarr(nlhs-1)
   gap = fltarr(nlhs-1)

   for i = 0, nlhs-2 do begin
      low1[i] = int_lowersemicircle_dx(r=r1, xc = x1, yc = y1, $
                                       x0 = xim[i], x1 = xim[i+1])
      low2[i] = int_lowersemicircle_dx(r=r2, xc = x2, yc = y2, $
                                       x0 = xim[i], x1 = xim[i+1])
      low3[i] = int_lowersemicircle_dx(r=r3, xc = x3, yc = y3, $
                                       x0 = xim[i], x1 = xim[i+1])
      high1[i] = int_uppersemicircle_dx(r=r1, xc = x1, yc = y1, $
                                        x0 = xim[i], x1 = xim[i+1])
      high2[i] = int_uppersemicircle_dx(r=r2, xc = x2, yc = y2, $
                                        x0 = xim[i], x1 = xim[i+1])
      high3[i] = int_uppersemicircle_dx(r=r3, xc = x3, yc = y3, $
                                        x0 = xim[i], x1 = xim[i+1])
      gap[i] = (eps_y/2.)*(xim[i+1] - xim[i])

   endfor
                                ; the total area is the upper half -
                                ; lower half integral, if it is
                                ; positive. If it is not positive,
                                ; that means that the lower curve is
                                ; above the upper curve, i.e. there is
                                ; no overlap region
   low = max([[low1], [low2], [low3]], dim = 2)
   high = min([[high1], [high2], [high3]], dim = 2)
   if total( finite(low, /nan) ) + total( finite(high, /nan)) gt 0. then stop ; consistency check
 
   dAp = ( ((high - gap) > 0.) - ((low - gap) > 0.) ) > 0.                     ; area segments in upper half plane
   dAm = ( ((high + gap) < 0.) - ((low + gap) < 0.) ) > 0.                     ; area segments in lower half plane
   dA = dAp + dAm
 
   area = area + total(dA)
   area_mp = area_mp + total(dAp)
   area_mm = area_mm + total(dAm)

endif

;stop
return, [area_pp, area_mp, area_pm, area_mm]

end



;------------------------------------------------------------
;------------------------------------------------------------
;
;
;                 GRID TRANSPARENCY FUNCTION            
;
;
;------------------------------------------------------------
;------------------------------------------------------------


function transparency_factor, phi, theta

; for the transparency model, we assume that the 
; eight grids are clocked evenly with respect to the 
; flow angle into the cup, i.e.

tanalpha = sqrt( tan(phi*!dtor)^2 + tan(theta*!dtor)^2 )
gridthick = 0.1d                 ; vertical thickness of grid
gridcell = 1d                   ; mm^2, actual
tau_0 = 0.888d                   ; transparent fraction
trans_area = tau_0*gridcell     ; transparent area of 1 cell
l = sqrt(trans_area)            ; lengthscale for transp part of cell
ngrids = 8d
d = (gridthick/l)*tanalpha

clockangles = dindgen(ngrids)/90d

; do this explicitly so that we can run the 
; procedure on an array of phi, theta inputs
t0 = tau_0*( 1d - d*sin(clockangles[0]*!dtor))*$
     (1d - d*cos(clockangles[0]*!dtor))
t1 = tau_0*( 1d - d*sin(clockangles[1]*!dtor))*$
     (1d - d*cos(clockangles[1]*!dtor))
t2 = tau_0*( 1d - d*sin(clockangles[2]*!dtor))*$
     (1d - d*cos(clockangles[2]*!dtor))
t3 = tau_0*( 1d - d*sin(clockangles[3]*!dtor))*$
     (1d - d*cos(clockangles[3]*!dtor))
t4 = tau_0*( 1d - d*sin(clockangles[4]*!dtor))*$
     (1d - d*cos(clockangles[4]*!dtor))
t5 = tau_0*( 1d - d*sin(clockangles[5]*!dtor))*$
     (1d - d*cos(clockangles[5]*!dtor))
t6 = tau_0*( 1d - d*sin(clockangles[6]*!dtor))*$
     (1d - d*cos(clockangles[6]*!dtor))
t7 = tau_0*( 1d - d*sin(clockangles[7]*!dtor))*$
     (1d - d*cos(clockangles[7]*!dtor))

return, t0*t1*t2*t3*t4*t5*t6*t7

end



;------------------------------------------------------------
;------------------------------------------------------------
;
;
;                 COLD RESPONSE FUNCTION            
;
;
;------------------------------------------------------------
;------------------------------------------------------------

function psp_swp_spc_coldspot, phi, theta

setparams, rl=rl, re=re, rc=rc, ze=ze, zl=zl, $
           epsilon=epsilon, dz_coll=dz_coll

if n_elements(phi) eq 0 then phi = 0.
if n_elements(theta) eq 0 then theta = 0.

mask = (cos(phi*!dtor) gt 0)*(cos(theta*!dtor ) gt 0)

; circle 1
r1 = re
x1 = ze*tan(phi*!dtor)
y1 = ze*tan(theta*!dtor)

; circle 2
r2 = rl
x2 = zl*tan(phi*!dtor)
y2 = zl*tan(theta*!dtor)

; circle 3
r3 = rc
x3 = 0d
y3 = 0d

eps_x = (epsilon - dz_coll * tan(abs(phi)*!dtor)) > 0.
eps_y = (epsilon - dz_coll * tan(abs(theta)*!dtor)) > 0.


areas = fltarr(4, n_elements(x1))
transp = transparency_factor(phi, theta)
for i = 0, n_elements(x1)-1 do $
   areas[*,i] = transp[i] * $
                quadrant_area_of_intersection(x1=x1[i], $
                   x2=x2[i], y1=y1[i], y2=y2[i], r1=r1, $
                   r2=r2, x3=x3, y3=y3, r3=r3, $
                   epsilon = [eps_x[i], eps_y[i]])
areas[0,*] = areas[0,*]*mask
areas[1,*] = areas[1,*]*mask
areas[2,*] = areas[2,*]*mask
areas[3,*] = areas[3,*]*mask
return, areas

end

function psp_swp_spc_coldspot_xyz, x, y, z
  if n_elements(z) eq 0 then z = 1.
  phi = atan(x, z)/!dtor
  theta = atan(y,z)/!dtor
  return, psp_swp_spc_coldspot(phi, theta)
  end
  


;------------------------------------------------------------
;------------------------------------------------------------
;
;
;                 WARM RESPONSE FUNCTIONS           
;
;
;------------------------------------------------------------
;------------------------------------------------------------



; solve effective area for a discrete distribution
; of flow angles described by the f(phi, theta)
; (would be more accurate with some kind of interpolation)
function distrospot_area, f, phi, theta, dphi, dtheta

areas = coldspot_areas(phi, theta)
norma = total(f*dphi*dtheta)
weights = f/norma
redim = size(phi, /dim)
if n_elements(redim) eq 1 then if redim eq 0 then redim = 1
a0 = total(reform(areas[0,*], redim)*weights*dphi*dtheta)
a1 = total(reform(areas[1,*], redim)*weights*dphi*dtheta)
a2 = total(reform(areas[2,*], redim)*weights*dphi*dtheta)
a3 = total(reform(areas[3,*], redim)*weights*dphi*dtheta)
return, [a0, a1, a2, a3]

end

; I should probably mod this so that it takes the phi, theta
; points as an argument. There is no reason to integrate this
; over (phi, theta) where sensitive area is zero (tan^2 phi + tan^2
; theta > tan^2 alphamax
;
function gaussian_spread_area, phi, theta, spread, num

; let's interpret that spread factor as the ratio of the cup 
; scale to the in-plane displacement, i.e. the 
; gaussian goes like exp( - ((tan phi - tan phi_0)^2 + (tan_theta -
;                           tan_theta_0)^2 )/ spread^2 )
;
; we'd like to get an evenly sampled grid that only spans the spot

n = 2.*fix(sqrt(float(num))/2.)

; remake these grids. Doesn't have to be as cute as I've been trying
pmin = atan(tan(phi*!dtor) - 4.*spread)/!dtor
pmax = atan(tan(phi*!dtor) + 4.*spread)/!dtor
tmin = atan(tan(theta*!dtor) - 4.*spread)/!dtor
tmax = atan(tan(theta*!dtor) + 4.*spread)/!dtor

p = pmin + (pmax - pmin)*findgen(n+1.)/n
t = tmin + (tmax - tmin)*findgen(n+1.)/n

pgrid = reform(p ## (1.+0.*t), (n+1.)^2 )
tgrid = reform( (1.+0.*p) ## t, (n+1.)^2 )
dp = 8.*spread/n
dt = 8.*spread/n
alphamax = 42.162
exfov = (tan(pgrid*!dtor)^2 + tan(tgrid*!dtor)^2) lt  tan(alphamax*!dtor)^2
exfov2 = (abs(pgrid) lt alphamax) and (abs(tgrid) lt alphamax)
tk = where((exfov*exfov2) eq 1, ntk)
if ntk le 1 then return, [0., 0., 0., 0.]
pgrid = pgrid[tk]
tgrid = tgrid[tk]
f = exp(  - ( (tan(phi*!dtor) - tan(pgrid*!dtor))^2 + (tan(theta*!dtor) - tan(tgrid*!dtor))^2) / (spread^2))
;stop
;plot, pgrid, tgrid, psym = 6, xrange = [-45, 45], yrange = [-45, 45]
;contour, f, pgrid, tgrid, /irreg, /follow, color = 254., /over, thick = 4
;stop
return, distrospot_area( f, pgrid, tgrid, dp, dt)

end


; subroutines for "intelligent" numerical integration of warm
; spot
function y_limits, x
common integration_constants, w, p0, t0, plims, tlims, alim
phi = atan(x)/!dtor
y0 = tan(t0*!dtor)
ymax = sqrt( tan(alim*!dtor)^2 - x^2)
threesig = [y0-3.*w, y0 + 3.*w]
return, [(-ymax) > threesig[0] , (ymax < threesig[1])]
end

; note that this function takes an array argument in the y 
; dim and a scalar in the x dim
function integrand0, x, y
common integration_constants, w, p0, t0, plims, tlims, alim
; working in the x-y plane to make normalization simpler
x0 = tan(p0*!dtor)
y0 = tan(t0*!dtor)
f = (1./(!dpi*w^2))*exp(  -( (x - x0)^2 + (y - y0)^2) / (w^2) )
areas = 0.*y
integrals = 0.*y
for i = 0, n_elements(y) - 1 do areas[i] = ((psp_swp_spc_coldspot( atan(x)/!dtor, atan(y[i])/!dtor ))[0])
return, f*areas
end
function integrand1, x, y
common integration_constants, w, p0, t0, plims, tlims, alim
; working in the x-y plane to make normalization simpler
x0 = tan(p0*!dtor)
y0 = tan(t0*!dtor)
f = (1./(!dpi*w^2))*exp(  -( (x - x0)^2 + (y - y0)^2) / (w^2) )
areas = 0.*y
integrals = 0.*y
for i = 0, n_elements(y) - 1 do areas[i] = ((psp_swp_spc_coldspot( atan(x)/!dtor, atan(y[i])/!dtor ))[1])
return, f*areas
end
function integrand2, x, y
common integration_constants, w, p0, t0, plims, tlims, alim
; working in the x-y plane to make normalization simpler
x0 = tan(p0*!dtor)
y0 = tan(t0*!dtor)
f = (1./(!dpi*w^2))*exp(  -( (x - x0)^2 + (y - y0)^2) / (w^2) )
areas = 0.*y
integrals = 0.*y
for i = 0, n_elements(y) - 1 do areas[i] = ((psp_swp_spc_coldspot( atan(x)/!dtor, atan(y[i])/!dtor ))[2])
return, f*areas
end
function integrand3, x, y
common integration_constants, w, p0, t0, plims, tlims, alim
; working in the x-y plane to make normalization simpler
x0 = tan(p0*!dtor)
y0 = tan(t0*!dtor)
f = (1./(!dpi*w^2))*exp(  -( (x - x0)^2 + (y - y0)^2) / (w^2) )
areas = 0.*y
integrals = 0.*y
for i = 0, n_elements(y) - 1 do areas[i] = ((psp_swp_spc_coldspot( atan(x)/!dtor, atan(y[i])/!dtor ))[3])
return, f*areas
end

; "intelligent" numerical integration of warm spot
; with some optimization
function psp_swp_spc_warmspot, phi, theta, spread, num=num;, num

; these settings were good in all testing that I tried
; should put an upper bound and do the hot limit as well
if spread lt 0.001 then return, psp_swp_spc_coldspot(phi, theta)
if n_elements(num) eq 0 then num = 20

setparams, rl=rl, re=re, rc=rc, ze=ze, zl=zl, epsilon=epsilon, $
  dz_coll=dz_coll, al=al, ac=ac, amax=amax

common integration_constants, w, p0, t0, plims, tlims, alim
p0=phi
t0=theta
w = spread
alim = amax
x0 = tan(p0*!dtor)
threesig = [x0-3.*w, x0 + 3.*w]
plims = [-amax, amax] ; should restrict this to within nsigma of center
xlims = [tan(plims[0]*!dtor) > threesig[0], $
         tan(plims[1]*!dtor) < threesig[1]]

; Using the function and limits defined above,
; unfortunately I don't have a workaround for doing this 4x
; at the moment
areas = [INT_2D('integrand0', xlims, 'y_limits', num, /DOUBLE), $
         INT_2D('integrand1', xlims, 'y_limits', num, /DOUBLE), $
         INT_2D('integrand2', xlims, 'y_limits', num, /DOUBLE), $
         INT_2D('integrand3', xlims, 'y_limits', num, /DOUBLE)]
return, areas

end



;------------------------------------------------------------
;------------------------------------------------------------
;
;
;                     TEST PROCEDURES             
;
;
;------------------------------------------------------------
;------------------------------------------------------------


; resolution test;
; I'm finding that the error goes like the number of points to
; the about -1.2 power. 
; Accuracy to 1 part in 1000 at around num = 1000.
; the comp time to get that accuracy (using a super dumb discrete
; method) is 
pro test3

phi = 6.
theta = 10.
spread = 1.
num = [4., 10., 20., 30., 40., 50., 100., 200., 250., 300., 250., 400., 250., 500., 550., 600., 750., 1000., 1200., 1500., 2000., 3000., 4000., 5000., 10000., 20000.]
cold = coldspot_areas(phi, theta)

t1 = systime(/seconds)
cold = coldspot_areas(phi, theta)
t2 = systime(/second)
tester = gaussian_spread_area(phi, theta, spread, 1000.)
t3 = systime(/second)
print, t3-t2
print, t2-t1 
; looks like the comp time for the warm solution is about 200x-300x
; slower. Here it's 650ms versus 3ms.
;
; I think we need this to work much faster for it to be deployable in
; a solver. There is surely a better/smarter way to do the convolution
warm = fltarr(4, n_elements(num))
for i = 0, n_elements(num)-1 do warm[*,i] = gaussian_spread_area(phi, theta, spread, num[i])
plot, num, abs(warm[0,*] - warm[0, n_elements(num)-1])/warm[0,n_elements(num)-1], /yl,  /xl, yrange = [1e-4, 1]
oplot, num, abs(warm[1,*] - warm[1, n_elements(num)-1])/warm[1,n_elements(num)-1]
oplot, num, abs(warm[2,*] - warm[2, n_elements(num)-1])/warm[2,n_elements(num)-1]
oplot, num, abs(warm[3,*] - warm[3, n_elements(num)-1])/warm[3,n_elements(num)-1]
oplot, num, num^(-1.2), color = 254., thick = 3
oplot, num, 4.*num^(-1.2), color = 254., thick = 3

stop
end

; this test compares the cold beam to a beam with a 20 degree half
; angle spread
pro test2

phis = findgen(60)
thetas = 0.*phis
cold = coldspot_areas(phis, thetas)
warm = 0.*cold
for i = 0, 59 do warm[*,i] = gaussian_spread_area(phis[i], thetas[i], 0.25, 1000.)
plot, phis, cold[0,*]
oplot, phis, warm[0,*], color = 254.

stop
end

pro test1, epsilon = epsilon, phi=phi, theta=theta

setparams, rl=rl, re=re, rc=rc, ze=ze, zl=zl, $ 
           dz_coll=dz_coll, al=al, ac=ac, amax=amax


if n_elements(phi) eq 0 then phi = 0.
if n_elements(theta) eq 0 then theta = 0.

; circle 1
r1 = re
x1 = ze*tan(phi*!dtor)
y1 = ze*tan(theta*!dtor)

; circle 2
r2 = rl
x2 = zl*tan(phi*!dtor)
y2 = zl*tan(theta*!dtor)

; circle 3
r3 = rc
x3 = 0d
y3 = 0d

dz_coll = 1.981
if n_elements(epsilon) eq 0 then epsilon = 0.4
eps_x = (epsilon - dz_coll * tan(abs(phi)*!dtor)) > 0.
eps_y = (epsilon - dz_coll * tan(abs(theta)*!dtor)) > 0.


; set up numerical computation
n = 1000.
xlims = [ ((x2-abs(r2)) > (x1-abs(r1))) > (x3-abs(r3)) ,  ((x2+abs(r2)) < (x1+abs(r1))) < (x3 + abs(r3)) ]
ylims = [ ((y2-abs(r2)) > (y1-abs(r1))) > (y3-abs(r3)) ,  ((y2+abs(r2)) < (y1+abs(r1))) < (y3 + abs(r3)) ]
xrange = xlims[1] - xlims[0]
yrange = ylims[1] - ylims[0]
dx = xrange/n
dy = yrange/n
xrays = (1.+fltarr(n)) ## (xlims[0] + dx*findgen(n)-dx/2.)
yrays = (ylims[0] + dy*findgen(n)-dy/2.) ## (1.+fltarr(n)) 
mask1 = ((xrays - x1)^2 + (yrays - y1)^2) le (r1^2)
mask2 = ((xrays - x2)^2 + (yrays - y2)^2) le (r2^2)
mask3 = ((xrays - x3)^2 + (yrays - y3)^2) le (r3^2)
mask = mask1*mask2*mask3
mask_pp = mask*(xrays gt (eps_x/2.))*(yrays gt (eps_y/2.))
mask_mp = mask*(xrays lt (-eps_x/2.))*(yrays gt (eps_y/2.))
mask_pm = mask*(xrays gt (eps_x/2.))*(yrays lt (-eps_y/2.))
mask_mm = mask*(xrays lt (-eps_x/2.))*(yrays lt (-eps_y/2.))
pass = where(mask eq 1., npass)
pp = where(mask_pp eq 1., npp)
mp = where(mask_mp eq 1., nmp)
pm = where(mask_pm eq 1., npm)
mm = where(mask_mm eq 1., nmm)
ingap = where( mask_pp eq 0. and mask_mp eq 0 and mask_pm eq 0 and mask_mm eq 0 and $
               mask eq 1, ngap)

temp = intersection_of_circles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
device, decomposed = 0
loadct, 0, /silent
plot, [x1-r1, x2-r2, x1+r1, x2+r2, x3-r3, x3+r3], [y1-r1, y1+r1, y2-r2, y2+r2, y3+r3, y3-r3], /nodata
oplot, xrays, yrays, color = 100, psym = 3
loadct, 0, /silent
if npass gt 0 then oplot, xrays[pass], yrays[pass], color = 50., psym = 3
loadct, 39, /silent
if npp gt 0 then oplot, xrays[pp], yrays[pp], color = 60., psym = 3
if nmp gt 0 then oplot, xrays[mp], yrays[mp], color = 120., psym = 3
if npm gt 0 then oplot, xrays[pm], yrays[pm], color = 180., psym = 3
if nmm gt 0 then oplot, xrays[mm], yrays[mm], color = 240., psym = 3
oplot, x1 + r1*cos(!pi*findgen(2000)/1000.), y1 + r1*sin(!pi*findgen(2000)/1000.)
oplot, x2 + r2*cos(!pi*findgen(2000)/1000.), y2 + r2*sin(!pi*findgen(2000)/1000.)
oplot, x3 + r3*cos(!pi*findgen(2000)/1000.), y3 + r3*sin(!pi*findgen(2000)/1000.)
oplot, temp[0] + [0,0], !y.crange
oplot, temp[1] + [0,0], !y.crange
oplot, !x.crange, [0,0]
oplot, [0,0], !y.crange
oplot, circle_xintercepts(xc=x1, yc=y1, r=r1), [0,0], psym = 6, thick = 2
oplot, circle_xintercepts(xc=x2, yc=y2, r=r2), [0,0], psym = 6, thick = 2
oplot, circle_xintercepts(xc=x3, yc=y3, r=r3), [0,0], psym = 6, thick = 2

areas = quadrant_area_of_intersection(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2, x3=x3, y3=y3, r3=r3, $
                                      epsilon = [eps_x, eps_y])
areas = [total(areas), areas]
numareas = [ npp*dx*dy, nmp*dx*dy, npm*dx*dy, nmm*dx*dy]
numareas = [total(numareas), numareas]
numgaparea = ngap*dx*dy

print, 'Analytical areas: ', string(areas, format = '(F7.3)') + [' = ', ' + ', ' + ', ' + ', '']
print, 'Numerical areas : ', string(numareas, format = '(F7.3)') + [' = ', ' + ', ' + ', ' + ', '']
print, 'N. gap loss area: ', string(numgaparea, format = '(F7.3)')
print, 'Num + N. gaploss: ', string(numgaparea + numareas[0], format = '(F7.3)')
print, 'Limiting ap area:  ', string( !dpi*double(min([r1, r2, r3]))^2, format = '(F9.5)')

stop


end

; calculate a new cold plasma table
pro test0

p = 0. + findgen(450.)/10.
t = 0. + findgen(450.)/10.
phi = p ## (1.+0.*t)
theta = (1.+0.*p) ## t

t1 = systime(/seconds)
cold = psp_swp_spc_coldspot(phi, theta)
t2 = systime(/seconds)
print, 'Calculated cold spot area ' + string(n_elements(phi)) + $
       ' times in ' + string(t2-t1) + ' seconds.'

dims = size(phi, /dim)
a0 = reform(cold[0,*], dims)
a1 = reform(cold[1,*], dims)
a2 = reform(cold[2,*], dims)
a3 = reform(cold[3,*], dims)
atot = a0+a1+a2+a3

cold = reform(cold, [4, size(phi, /dim)])
;contour, a0+a1+a2+a3, tan(phi*!dtor), tan(theta*!dtor), /iso
loadct, 39
device, decomposed = 0
contour, a0, tan(phi*!dtor), tan(theta*!dtor), /iso, xtitle = 'tan(phi)', ytitle = 'tan(theta)', $
         title = 'cold effective area', charsize = 2, levels = max(a0)*[0.01, 0.2, 0.4, 0.6, 0.8], /nodata
oplot, !x.crange, [0,0]
oplot, [0,0], !y.crange
contour, atot, tan(phi*!dtor), tan(theta*!dtor), /iso, /over
contour, a1, tan(phi*!dtor), tan(theta*!dtor), /iso, /over, color = 100., levels = max(a0)*[0.01, 0.2, 0.4, 0.6, 0.8]
contour, a2, tan(phi*!dtor), tan(theta*!dtor), /iso, /over, color = 150., levels = max(a0)*[0.01, 0.2, 0.4, 0.6, 0.8]
contour, a3, tan(phi*!dtor), tan(theta*!dtor), /iso, /over, color = 50., levels = max(a0)*[0.01, 0.2, 0.4, 0.6, 0.8]
contour, a0, tan(phi*!dtor), tan(theta*!dtor), /iso, /over, color = 254., thick = 4, levels = max(a0)*[0.01, 0.2, 0.4, 0.6, 0.8]
oplot, tan(42.162*!dtor) * cos(findgen(360)*!dtor), $
       tan(42.162*!dtor) * sin(findgen(360)*!dtor), thick = 5

x = ((a0 + a1) - (a2+a3))/(atot > 1e-4)
y = ((a0 + a2) - (a1+a3))/(atot > 1e-4)

; now it would be good to make a comparison with the old version


stop

end

pro timetest

phi = 90.*randomn(seed, 20)
theta = 90.*randomn(seed, 20)
t1 = systime(/sec)
for i = 0, n_elements(phi) do temp = psp_swp_spc_warmspot(phi[i], theta[i], 0.2)
t2 = systime(/sec)
print, (t2-t1)/n_elements(phi)

; looks like we're at 600 ms per warm spot evaluation.
; that's about the same as the dumb grid approach, unfortunately

end

pro timetest2, phi, theta

t1 = systime(/sec)
a = psp_swp_spc_coldspot(phi, theta)
t2 = systime(/sec)
print, t2-t1

; if warm calculations are 300x slower, 
; the warm table will take about 6.25 hours per mach number
; to populate. If we run 33 machs, that will take 8.5 days
; 
; if we cut that down to the wedge, it will be about 45 minutes per
; mach number (one day). Seems worth launching into that now
;
; if we could figure out a way to do our integrator in parallel
; instead of four times in series, we could get that down to 12 minutes

stop

end


pro tablegen

restore, '/psp/code/calibration/spp_swp_spc_calfiles_spc_uvphithetalookup_20190819.idl'

; get range vals for table
phi = table.phi
theta = table.theta
mach = table.machnum
spreads = 1./mach

; clear memory. That's a big table after all
table = -1

; limit calculations to the 45 degree wedge from clock angle 0 to 45
clock = atan( tan(theta*!dtor), tan(phi*!dtor) )/!dtor
tk = where(clock ge 0 and clock le 45.)
phi = phi[tk]
theta = theta[tk]

areas = fltarr(4, n_elements(phi), n_elements(spreads))
steptimes = fltarr(n_elements(phi)) + !values.f_nan

restore, '/home/mstevens/temp_spctable.idl'
j0 = j

for j = j0, n_elements(phi)-1 do begin
   print, 'Angle ', j, ' of ', n_elements(phi)
   print, 'Solving phi = ', phi[j], ' theta = ', theta[j]
   t1 = systime(/sec)
   for i = 0, n_elements(spreads)-1 do areas[*, j,i ] = psp_swp_spc_warmspot(phi[j], theta[j], spreads[i])
   t2 = systime(/sec)
   steptimes[i] = t2-t1
   tav = mean(steptimes, /nan)
   print, '...done, took ', t2-t1, ' seconds'
   print, 'Est time to completion ', tav*float(n_elements(phi) - i)/(60.*60.), ' hours.'
   if j mod 10 eq 0 then save, phi, theta, mach, i, j, areas, filename = '/home/mstevens/temp_spctable.idl'
endfor

end

      
    


pro tablege_piece, jmin = jmin

jminstr = strtrim(string(jmin),1)

restore, '/psp/code/calibration/spp_swp_spc_calfiles_spc_uvphithetalookup_20190819.idl'

; get range vals for table
phi = table.phi
theta = table.theta
mach = table.machnum
spreads = 1./mach

; clear memory. That's a big table after all
table = -1

; limit calculations to the 45 degree wedge from clock angle 0 to 45
clock = atan( tan(theta*!dtor), tan(phi*!dtor) )/!dtor
tk = where(clock ge 0 and clock le 45.)
phi = phi[tk]
theta = theta[tk]

areas = fltarr(4, n_elements(phi), n_elements(spreads)) + !values.f_nan
steptimes = fltarr(n_elements(phi)) + !values.f_nan

;restore, '/home/mstevens/temp_spctable.idl'
j0 = j

for j = tmin, n_elements(phi)-1 do begin
   print, 'Angle ', j, ' of ', n_elements(phi)
   print, 'Solving phi = ', phi[j], ' theta = ', theta[j]
   t1 = systime(/sec)
   for i = 0, n_elements(spreads)-1 do areas[*, j,i ] = psp_swp_spc_warmspot(phi[j], theta[j], spreads[i])
   t2 = systime(/sec)
   steptimes[i] = t2-t1
   tav = mean(steptimes, /nan)
   print, '...done, took ', t2-t1, ' seconds'
   print, 'Est time to completion ', tav*float(n_elements(phi) - i)/(60.*60.), ' hours.'
   if j mod 10 eq 0 then save, phi, theta, mach, i, j, areas, filename = '/home/mstevens/temp_spctable_'+jminstr+'.idl'
endfor

end

