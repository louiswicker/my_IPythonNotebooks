""" FORMAT: Field: ['Formal Name', 'Units', Positive, Scale, Min Contour] 
            
            Field:        String  Variable name read from file
            
            Formal Name:  String  Actual name of field
            Units:        String  Formatted string used to display 
                                  field's units
            X-Axis Label: String  Default title for X-axis
            Y-Axis Label: String  Default title for Y-axis
            Positive:     Bool    True if field is positive definite
            Scale:        Float   Default scale value of field
            Min Contour
                Interval: Float   Minimum contour interval allowed for field
            Min Value:    Float   Minium value to put into field for quick contouring
            Max Value:    Float   Maxium value to put into field for quick contouring """
fields = {
    'U': ['Wind U-Component', '$m^{}s^{-1}$', 'X (km)', 'Y (km)', False, 1., 2.0, -30., 30.],
    'V': ['Wind V-Component', '$m^{}s^{-1}$', 'X (km)', 'Y (km)', False, 1., 2.0, -30., 30.],    
    'W': ['Vertical Velocity', '$m^{}s^{-1}$', 'X (km)', 'Y (km)', False, 1., 1.0, -10., 10.],
    'PI': ['Pert. Pressure', '$hPa$', 'X (km)', 'Y (km)', False, 1., 1.0, -20., 20.],
    'KM': ['Diagnostic Mixing', '$m^{2^{}}s^{-2}$', 'X (km)', 'Y (km)', True, 1., 10., 0.0, 1000.],
    'TH': ['Pert. Pot. Temperature', '$K$', 'X (km)', 'Y (km)', False, 1., 0.5, -10., 10.],
    'QV': ['Pert. Water Vap. Mass', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', False, 1000., 1.0, 0.0, 10.],
    'QR': ['Rainwater Mass', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QC': ['Cloud Water Mass', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QI': ['Cloud Ice Mass', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QH': ['Hail/Graupel Mass', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QS': ['Snow Mass', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'DBZ': ['Reflectivity', '$dBZ$', 'X (km)', 'Y (km)', True, 1., 5., 0.0, 70.],
    'WZ': ['Vertical Vorticity', '$10^{-4}s^{-1}$', 'X (km)', 'Y (km)', False, 10000., 10., -300., 300.],
    'QGL': ['Graupel - Low density', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QGM': ['Graupel - Med. density', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QGH': ['Graupel - High density', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QF': ['Frozen Drops', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QHL': ['Large Hail', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QIP': ['Ice plates', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QIR': ['Rimed Ice', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QGTOT': ['Total Graupel/Hail', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QICE': ['Total Ice', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QTOT': ['Total hydrometeor', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'QLIQ': ['Total Liquid', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1000., 1.0, 0.0, 10.],
    'THV': ['Pert. Virt. Pot. Temp.', '$K$', 'X (km)', 'Y (km)', False, 1., 1.0, -10., 10.],
    'RAIN_ACC': ['Accumulated Rain', '$g^{}kg^{-1}$', 'X (km)', 'Y (km)', True, 1., 0.1, 0.0, 5.0],
    'CCCN': ['CCN Concentration', '$cm^{-1}$', 'X (km)', 'Y (km)', True, 1e-6, 0.01, 0.0001, 0.1],
    'CRW': ['Rain Conc.', '$L^{-1}$', 'X (km)', 'Y (km)', True, 0.001, 0.01],
    'CCW': ['Cloud Droplet Conc.', '$cm^{-1}$', 'X (km)', 'Y (km)', True, 1e-6, 0.01],
    'CCI': ['Cloud Ice Conc.', '$L^{-1}$', 'X (km)', 'Y (km)', True, 0.001, 0.01],
    'CHW': ['Graupel Conc.', '$L^{-1}$', 'X (km)', 'Y (km)', True, 0.001, 0.01],
    'CSW': ['Snow Conc.', '$L^{-1}$', 'X (km)', 'Y (km)', True, 0.001, 0.01],
    'CHL':        ['Hail Conc.', '$L^{-1}$', 'X (km)', 'Y (km)', True, 0.001, 0.01],
    'WIND_SPEED': ['Wind Speed', '$m^{}s^{-1}$', 'X (km)', 'Y (km)', True, 1, 5.0, 0.0, 50.],
    'UH': ['Updraft Helicity', '$m^{2}s^{-2}$', 'X (km)', 'Y (km)', True, 1, 25.0, 0.0, 500.],
    'VR': ['Radial Velocity', '$m^{}s^{-1}$', 'X (km)', 'Y (km)', False, 1, 5.0, -40., 40.]}
