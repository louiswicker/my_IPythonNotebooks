!------------------------------------------------------------------------------
!
!   /////////////////////          BEGIN         \\\\\\\\\\\\\\\\\\\\\\
!   \\\\\\\\\\\\\\\\\\\\\   SUBROUTINE KESSLER   //////////////////////
!
! KESSLER does a 3 water catagory microphysical parameterization
!
! Created by:  Louis Wicker, July 18, 1988
! Latest Revision: 06-10-00
!
! Update Notes:
!------------------------------------------------------------------------------
 subroutine kessler(t,         &	 ! potential temperature
                    qv,		      &		! water vapor mixing ratio
                    qc, 	      &  ! cloud water mixing ratio at t 
                    qr,        &		! rain  water mixing ratio at t 
                    tb, pb,	   &		! base state theta and pi
                    dt, 	      &		! time step
                    dz, 	      &		! grid spacing
                    nz)	      		  ! grid dimension

!------------------------------------------------------------------------------

!------------------------------------------------------------------------------

	real t00     ;    parameter( t00    = 300.0     )
	real p00     ;    parameter( p00    = 1.0e5     )
 real rgas    ;    parameter( rgas   = 287.04    )
 real rv      ;    parameter( rv     = 461.00    )
 real cp      ;    parameter( cp     = 1004.     )
 real cpv     ;    parameter( cpv    = 1870.     )
 real Lv      ;    parameter( Lv     = 2.5e6     )
 real Cl      ;    parameter( Cl     = 4190.     )
 real epsilon ;    parameter( epsilon= 0.622     )

! Passed variables

 integer, intent(IN) :: nz
 real(kind=8), intent(INOUT), dimension(nz) ::  t, qv, qc, qr, tb, pb
	real(kind=8), INTENT(IN) ::  dt, dz

! Local variables

	real(kind=8) qrprod(nz), prod(nz+1)
	real(kind=8) rcgs(nz+1), vt(nz+1)
	real(kind=8) pinit(nz)
 real(kind=8) gam(nz)

! Local constants for the microphysics parameterization
 real(kind=8) c1, c2, c3, c4
 real(kind=8) prod0, ern, temp
 parameter( c1 = .001, c2 = .001, c3 = 2.2, c4 = .875 )
 parameter( f5 = 237.3 * 17.27 * 2.5e6 / cp )

!------------------------------------------------------------------------------

  DO k = 1,nz

   rcgs(k)   = 1.0e2*pb(k)**2.509/(287.04*tb(k))
   pinit(k)  = 1.0e5*pb(k)**3.509
   gam(k)    = Lv/(Cp*pb(k))

  ENDDO
  rcgs(nz+1)   = rcgs(nz)

! Compute autoconversion, coalesense, and set tmp arrays for time split
! terminal velocity fall.

  DO k = 1,nz
	   factor    = 1.0 / (1.+c3*dt*qr(k)**c4)
    qrprod(k) = qc(k) * (1.0 - factor) + factor*c1*dt*max(qc(k)-c2,0.)      
  ENDDO

! Copy qr array
! Create fall velocities

  DO k = 1,nz
    prod(k) = qr(k)
!   vt(k)   = 36.34*(qr(k)*rcgs(k))**0.1364 * sqrt(1.1225D-3/rcgs(k))
    vt(k)   = 31.25*(qr(k)*rcgs(k))**0.125
  ENDDO
  vt(nz+1)   = 0.0
  prod(nz+1) = 0.0

! Fallout done with flux upstream

  DO k = 1,nz

    factor = dt/(dz*rcgs(k))
    prod(k) = prod(k) - factor              &
            * (rcgs(k  )*prod(k  )*vt(k  )  &
              -rcgs(k+1)*prod(k+1)*vt(k+1))

  ENDDO

! Production of rain and deletion of qc
! Production of qc from supersaturation
! Evaporation of QR

  DO k = 1,nz

    qc(k) = max(qc(k)   - qrprod(k),0.)
    qr(k) = max(prod(k) + qrprod(k),0.)
    temp  = pb(k) * t(k)
    qvs   = 380.*exp(17.27*(temp-273.)/(temp- 36.))/pinit(k)
    prod0 = (qv(k)-qvs) / (1.0D0+qvs*f5/(temp-36.)**2)
    ern   = dmin1(dt*(((1.6D0+124.9D0*(rcgs(k)*qr(k))**.2046) &
            *(rcgs(k)*qr(k))**.525)/(2.55D8/(pinit(k)*qvs)    &
              +5.4D5))*(dim(qvs,qv(k))/(rcgs(k)*qvs)),        &
             dmax1(-prod0-qc(k),0.0D0),qr(k))

! Next line shuts off conversion at low temperatures - keeps qc up at anvil level
! This was used in storm video - makes a pretty picture

	   IF(prod0 .lt. 0.0 .and. temp .lt. 253.) prod0 = 0.0

! Update all variables

    product = max(prod0,-qc(k))
    t (k) = t(k) + gam(k)*(product - ern)
    qv(k) = max(qv(k) - product + ern,0.)
    qc(k) =     qc(k) + product
    qr(k) = qr(k) - ern

  ENDDO

  RETURN  
  END SUBROUTINE KESSLER
