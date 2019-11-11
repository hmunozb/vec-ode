

pub mod quad{
    pub static C_GAUSS_LEGENDRE_4: [f64; 2] = [0.21132486540518711775, 0.78867513459481288225 ];
    //static C_GAUSS_LEGENDRE_6: [f64; 3] =[];
}

pub mod rk{
    pub static rk45_ac : [f64; 36] = [
        0.,                     0.,0.,0.,0.,0.,
        1./4.,
        1./4.,                  0.,0.,0.,0.,
        3.0/32.,     9.0/32.,
        3./8.,                  0.,0.,0.,
        1932./2197., -7200./2197., 7296./2197.,
        12./13.,                0.,0.,
        439./216.,   -8.,          3680./513.,   -845./4104.,
        1.0,                    0.,
        -8./27.,     2.,           -3544./2526., 1859./4104.,    -11./40.,
        1.0/2.0 ];

    pub static rk45_b: [f64; 6] = [
        16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.];

    pub static rk45_berr: [f64; 6] = [
        25./216.,0.,1408./2565.,2197./4104., -1./5., 0.
    ];
}

pub mod split{
    pub static RKN_O4_A : [f64; 3] = [
        0.209515106613362, -0.143851773179818, 0.434336666566456
    ];

    pub static RKN_O4_B : [f64; 4] = [
        0.0792036964311957, 0.353172906049774, -0.0420650803577195, 0.21937695575349958
    ];
}

pub mod split_complex {
    use num_complex::Complex64 as c64;

    pub static TJ_O4_A:[c64; 2] = [
        c64{re:0.32439640402017118298, im: 0.13458627249080669679},
        c64{re: 0.35120719195965763405, im: -0.26917254498161339358}
    ];

    pub static TJ_O4_B:[c64; 2] = [
        c64{re: 0.16219820201008559149, im: 0.06729313624540334839},
        c64{re: 0.33780179798991440851, im: -0.06729313624540334839}
    ];

    pub static SEMI_COMPLEX_O4_A: [c64; 2] =
        [c64{re: 0.25, im: 0.0}, c64{re: 0.25, im: 0.0}];

    pub static SEMI_COMPLEX_O4_B: [c64; 3] =
        [   c64{re: 0.1, im: -1.0/30.0},
            c64{re: 4.0/15.0, im: 2.0/15.0},
            c64{re: 4.0/15.0, im: -1.0/15.0}];

}

pub mod cfqm {
    pub static CFM_R2_J1_GL: [f64; 2] = [
        0.5, 0.5
    ];

    pub static CFM_R4_J2_GL: [f64; 4] = [
        -0.038675134594812882255, 0.53867513459481288225,
        0.53867513459481288225, -0.038675134594812882255
    ];

    pub static BLANES17_R4_J4: [f64; 12] = [
        0.2463347584748155,     -0.0469610812011527,    0.0119511881315244,
        0.0622500005170514,     0.2691833034233750,     -0.0427581693456134,
        -0.0427581693456134,    0.2691833034233750,     0.0622500005170514,
        0.0119511881315244,     -0.0469610812011527,    0.2463347584748155];

}