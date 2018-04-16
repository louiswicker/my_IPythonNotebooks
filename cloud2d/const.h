	real t00     ;    parameter( t00    = 300.0     )
	real p00     ;    parameter( p00    = 1.0e5     )
      real rgas    ;    parameter( rgas   = 287.04    )
      real rv      ;    parameter( rv     = 461.00    )
      real cp      ;    parameter( cp     = 1004.     )
      real cpv     ;    parameter( cpv    = 1870.     )
      real Lv      ;    parameter( Lv     = 2.5e6     )
      real Cl      ;    parameter( Cl     = 4190.     )
      real epsilon ;    parameter( epsilon= 0.622     )
      real cthres  ;    parameter( cthres = 1.0e-6    )
      real Cm      ;    parameter( Cm     = 0.21      )
      real Pr      ;    parameter( Pr     = 1.5       )
      real g       ;    parameter( g      = 9.8106    )
      real Cv      ;    parameter( cv     = 717.      )
      real Cs      ;    parameter( cs     = 150.      )
	real pii     ;    parameter( pii    = 3.1415926 )
      real gcp     ;    parameter( gcp    = g / cp    )
      real rcp     ;    parameter( rcp    = rgas / cp )
      real cvr     ;    parameter( cvr    = cv / rgas )
      real alpha   ;    parameter( alpha  = 0.25      )
      real dxt     ;    parameter( dxt    = 40.0      )
	real lapse   ;    parameter( lapse  = 0.0       )
	real kdiv0   ;    parameter( kdiv0  = 0.05      )

! Dont mess with stuff above this line unless you know what you are doing!

! Parameters controling mixing parameterization

      integer mix_type  ;    parameter( mix_type  = -1    ) ! [-1,0]   => [ahighk,Lilly]
      integer len_type  ;    parameter( len_type  = 0     ) ! [0,1,2] => [volume,dz,BLR]
      real    Lmax      ;    parameter( Lmax      = 500.  )
      real    ahighk    ;    parameter( ahighk    = 75. )

! Parameters controling Rayeigh damper

	real    rayd_hgt  ;    parameter( rayd_hgt  = 12000.)
	real    rayd_mag  ;    parameter( rayd_mag  = 0.005 )

! Boundary condition parameters (periodic not implemented yet...)

	integer bcx;    parameter(bcx = -1) ! [-1,0,1] -> [periodic,rigid,open] 
	integer bcz;    parameter(bcz =  0)  ! [ 0,1  ] -> [FS/rigid, semi-slip & rigid]
      real drag  ;    parameter( drag   = 0.003     )
