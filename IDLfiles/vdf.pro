; 
; vdf object class
; -----------------
; this is a base class for looking up velocity 
; distribution function values. 
;
; METHOD vdf.evaluate(arg)
; the vdf.evaluate(arg) function returns evaluations 
; of the vdf, where are is assumed to be a 3-vector or 
; array of 3-vectors 
;
; METHOD vdf.set_func(arg)
; assigns the evaluation method to the function of 
; choice, where are is the [string] name of the function
;
; METHOD vdf.set_udata(arg)
; stores any support data that is necessary for 
; calling the evaluation function
;
; two sample implimentations are included: 
; 'empty_vdf' - returns 0 at all locations
; 'maxwellian_vdf' -- evaluates a 3D maxwellian, assuming 
;                     that the udata supplies a 
;                     structure with tags {vx, vy, vz, w, n}

function empty_vdf, vxyz, _extra
  return, 0.*fltarr(n_elements(vxyz)/3)
end

function maxwellian_vdf, vxyz, params_ptr
  params = *params_ptr
  result = empty_vdf(vxyz)
  if n_elements(params) lt 1 then return, result
  vecdim = where(size(vxyz, /dim) eq 3)

  if vecdim[0] eq 0 then begin
    vx = reform(vxyz[0,*])
    vy = reform(vxyz[1,*])
    vz = reform(vxyz[2,*])
  endif else begin
    vx = reform(vxyz[*,0])
    vy = reform(vxyz[*,1])
    vz = reform(vxyz[*,2])
  endelse

  ux = params.vx
  uy = params.vy
  uz = params.vz
  w = params.w
  n = params.n

  g = ( params.n/(sqrt(!pi)*params.w)^3 ) * exp( -( (vx-ux)^2 + (vy-uy)^2 + (vz-uz)^2 )/(params.w^2) )
  return, g

end


; Make a class for VDFs so that we can call the evaluation in the integral
function vdf::evaluate, vxyz
  return, call_function(self.func_name, vxyz, self.udata)
end

pro vdf::set_function, func_name
  if n_elements(func_name) ne 0 then self.func_name = func_name
  return
end

pro vdf::set_udata, udata
  if n_elements(udata) ne 0 then begin
    ptr_free, self.udata
    self.udata=ptr_new(udata, /allocate)
  endif
  return
end

function vdf::init
  return, 1
end

function vdf::cleanup
  ptr_free, self.udata
  return, 1
end

pro vdf__define, params
  void = {vdf, func_name:'', udata:ptr_new()}
  return
end

function new_vdf, func_name, udata
  a = obj_new('vdf')
  a.set_udata, udata
  a.set_function, func_name
  return, a
end

pro vdf_test
  f = new_vdf('empty_vdf')
  print, f.evaluate( [0, 0, 0] )

  ; would be more orderly to extend vdf with maxwellian_vdf and then
  ; have these parameters as object vars instead of in the pointer
  ; because maxwellian_vdf will break now if you don't give it the
  ; right parameter format. Regardless...
  g = new_vdf('maxwellian_vdf', {vx:100., vy:0., vz:0., w:50., n:100.})
  print, g.evaluate( [[80, 0, 0], [90, 0, 0], [100, 0, 0], [110, 0, 0]])

end
