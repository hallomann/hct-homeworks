import numpy as np
cimport numpy as np
from libc.math cimport exp, log, pow, fabs

cdef double R = 8.314472
cdef double T = 310.5
cdef double F = 96.4845
cdef int number_of_apds = 100

cdef double ddt
cdef double total_time
cdef double capacitance = 0.025
cdef double vcell = 3.0
cdef double l_cell = 66.3767257
cdef double r_cell = 3.792956
cdef double vrel = 0.0036
cdef double vsub = 0.03328117
cdef double vup = 0.0348
cdef double vi = 1.34671883
cdef double Mgi = 2.5
cdef double nao = 140.0
cdef double cao = 1.8
cdef double ko = 5.4

cdef double gcal12 = 0.0010 * 4.0 * 5
cdef double gcal13 = 0.0030 * 4.0 * 5

cdef double gbna = 2.5 * 0.0001215  #
cdef double gk1 = 0.1 * 0.229 * 0.0039228 * 0.9  #
cdef double gcat = 0.75 * 0.01862
cdef double gto = 0.00492
cdef double gh = 0.0057

cdef double kNaCa = 5.5

cdef double gst = 0.00006
cdef double eist = 17.0

cdef double gbca = 0.000015
cdef double gbk = 0.0000025

cdef double gks = 0.000299
cdef double gkr = 0.8 * 0.002955

cdef double ecal = 47.0
cdef double kmfca = 0.00035
cdef double alpha_fca = 0.021
cdef double all_ica_multiplier = 1.0
cdef double ecat = 45.0
cdef double enattxr = 41.5761
cdef double multiplier2 = 1.0
cdef double gsus = 0.00039060
cdef double inakmax_multiplier = 1.85
cdef double inakmax = inakmax_multiplier * 0.077

cdef double gna_ttxs = 0.1 * 5.925e-05
cdef double gna_ttxr = 0.1 * 5.925e-05

cdef double kmnap = 14.0
cdef double kmkp = 1.4
cdef double K1ni = 395.3
cdef double K1no = 1628
cdef double K2ni = 2.289
cdef double K2no = 561.4
cdef double K3ni = 26.44
cdef double K3no = 4.663
cdef double Kci = 0.0207
cdef double Kco = 3.663
cdef double Kcni = 26.44
cdef double Qci = 0.1369
cdef double Qco = 0.0
cdef double Qn = 0.4315
cdef double tdifca = 0.04
cdef double Prel = 2.5
cdef double Krel = 0.0015
cdef double nrel = 2.0
cdef double Kup = 0.0006
cdef double nup = 1.0
cdef double Ttr = 40.0
cdef double ConcTC = 0.031
cdef double ConcTMC = 0.062
cdef double kfTC = 88.8
cdef double kfTMC = 237.7
cdef double kbTC = 0.446
cdef double kbTMC = 0.00751
cdef double kfTMM = 2.277
cdef double kbTMM = 0.751
cdef double ConcCM = 0.045
cdef double kfCM = 237.7
cdef double kbCM = 0.542
cdef double ConcCQ = 10.0
cdef double kfCQ = 0.534
cdef double kbCQ = 0.445
cdef double koca = 10.0
cdef double kom = 0.06
cdef double kica = 0.5
cdef double kim = 0.005
cdef double eca50sr = 0.45
cdef double maxsr = 15.0
cdef double minsr = 1.0
cdef double hsrr = 2.5
cdef double pumphill = 2.0

cdef double v = -64.5216286940
cdef double dst = 0.6246780312
cdef double fst = 0.4537033169
cdef double dt = 0.0016256324
cdef double ft = 0.4264459666
cdef double ikr_act = 0.4043600437
cdef double ikr_inact = 0.9250035423
cdef double ikr_inact2 = 0.1875749806
cdef double iks_act = 0.0127086259
cdef double fl12 = 0.9968141226
cdef double dl12 = 0.0000045583
cdef double fl13 = 0.9809298233
cdef double dl13 = 0.0002036671
cdef double r = 0.0046263658
cdef double m_ttxr = 0.4014088304
cdef double h_ttxr = 0.2724817537
cdef double j_ttxr = 0.0249208708
cdef double m_ttxs = 0.1079085266
cdef double h_ttxs = 0.4500098710
cdef double j_ttxs = 0.0268486392
cdef double y_1_2 = 0.0279984462
cdef double y_4 = 0.0137659036
cdef double carel = 0.1187281829
cdef double caup = 1.5768287365
cdef double casub = 0.0000560497
cdef double Ftc = 0.0063427103
cdef double Ftmc = 0.1296677919
cdef double Ftmm = 0.7688656371
cdef double Fcms = 0.0242054739
cdef double Fcmi = 0.0138533048
cdef double Fcq = 0.1203184861
cdef double cai = 0.0000319121
cdef double q = 0.6107148187
cdef double fca = 0.7649576191
cdef double nai = 8.1179761505
cdef double ki = 139.8854603066
cdef double resting = 0.7720290515
cdef double openg = 0.0000000760
cdef double inactivated = 0.0000000213
cdef double resting_inactivated = 0.2162168926
cdef double current_trace_counter = 0
cdef int output_counter = 0
cdef double variation_multiply = 1.0
cdef int number = 0

cdef np.ndarray[np.float64_t, ndim=1] min_potential = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] tmin_potential = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] max_potential = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] dvdtmax = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] vdvdtmax = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] apd_start = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] apd_end = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] cai_peak = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] cai_min = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] ddr = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] top = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] top_slope = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] apd50 = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] apd90 = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] cycle_length = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] casub_peak = np.zeros(number_of_apds)
cdef np.ndarray[np.float64_t, ndim=1] casub_min = np.zeros(number_of_apds)

cdef double dummy = 1.0
cdef double base_cycle = 0.0
cdef double pumpvmax = 0.04
cdef double pumpkmf = 0.00008
cdef double pumpkmr = 4.5
cdef double pumpfactor = 1.0

cdef int param_counter = 0
cdef double dvdt = 1000.0
cdef double dvdtold = 500.0
cdef int start_output = 0

cdef double FRT = F / (R * T)

cdef double vhalf_gh = 106.8
cdef double gkr = 0.8 * 0.002955
cdef double gto = 0.00492
cdef double kNaCa = 5.5

cdef double Pup = 20 * 0.04  #
cdef double ks = 1300000.0

cdef double[:] t
cdef double[:] VV

def Kharche(double ddt, double total_time):
    global v, dst, fst, dt, ft, ikr_act, ikr_inact, ikr_inact2, iks_act, fl12, dl12, fl13, dl13, r, m_ttxr, h_ttxr, j_ttxr, m_ttxs, h_ttxs, j_ttxs, y_1_2, y_4, carel, caup, casub, Ftc, Ftmc, Ftmm, Fcms, Fcmi, Fcq, cai, q, fca, nai, ki, resting, openg, inactivated, resting_inactivated, current_trace_counter, output_counter, variation_multiply, number
    global min_potential, tmin_potential, max_potential, dvdtmax, vdvdtmax, apd_start, apd_end, cai_peak, cai_min, ddr, top, top_slope, apd50, apd90, cycle_length, casub_peak, casub_min
    global dummy, base_cycle, pumpvmax, pumpkmf, pumpkmr, pumpfactor
    global param_counter, dvdt, dvdtold, start_output
    global FRT, vhalf_gh, gkr, gto, kNaCa, Pup, ks

    cdef double sstime
    cdef double ena, ek, eks, eca
    cdef double qa, alphaqa, betaqa, tauqa, alphaqi, betaqi, qi, tauqi
    cdef double alpha_dl, beta_dl, tau_dl, dl13_inf, fl13_inf, tau_fl, fl13, dl13, dl12_inf, fl12_inf, dl12, fl12, fca_inf, taufca, fca
    cdef double fna, m3_inf_ttxr, h_inf_ttxr, m3_inf_ttxs, h_inf_ttxs, m_inf_ttxr, m_inf_ttxs, tau_m, tau_h, tau_j, m_ttxs, h_ttxs, j_ttxs, hs, tau_mr, tau_hr, tau_jr, m_ttxr, h_ttxr, j_ttxr, hsr
    cdef double ina_ttxs, ina_ttxr
    cdef double y_inf, tau_y_1_2, y_1_2, ihk, ihna, ih
    cdef double q_inf, tau_q, q, r_inf, tau_r, r, ito
    cdef double isus, inak, inaca
    cdef double di, doo, k43, k12, k14, k41, k34, k21, k23, k32, x1, x2, x3, x4
    cdef double ca_flux, Jcadif, kcasr, kosrca, kisrca, resting, openg, inactivated, resting_inactivated, Jrel, Jup, Jtr, dFtc, dFtmc, dFtmm, dFcms, dFcmi, dFcq, Ftc, Ftmc, Ftmm, Fcms, Fcmi, Fcq, casub, cai, carel, caup, dvdtold, total_current, dvdt, vnew, ena, ek, eks, eca, nai_tot, ki_tot, nai, ki

    for param_counter in range(number_of_apds):
        min_potential[param_counter] = 100000.0
        max_potential[param_counter] = -10000.0
        dvdtmax[param_counter] = -10000.0
        ddr[param_counter] = -10000.0
        top[param_counter] = 10000.0
        top_slope[param_counter] = -10000.0
        apd50[param_counter] = -10000.0
        apd90[param_counter] = -10000.0
        cycle_length[param_counter] = -10000.0

    ena = (R * T / F) * log(nao / nai)
    ek = (R * T / F) * log(ko / ki)
    eks = ((R * T) / F) * log((ko + 0.12 * nao) / (ki + 0.12 * nai))
    eca = (R * T / (2 * F)) * log(cao / casub)

    t = np.zeros(int(total_time / ddt))
    VV = np.zeros(int(total_time / ddt))

    for param_counter in range(int(total_time / ddt)):
        sstime = param_counter * ddt

        qa = 1.0 / (1.0 + exp(-(v + 67.0) / 5.0))
        alphaqa = 1.0 / (0.15 * exp(-(v) / 11.0) + 0.2 * exp(-(v) / 700.0))
        betaqa = 1.0 / (16.0 * exp((v) / 8.0) + 15.0 * exp((v) / 50.0))
        tauqa = 1.0 / (alphaqa + betaqa)
        alphaqi = (
            0.15 * 1.0 / (3100.0 * exp((v + 10.0) / 13.0) + 700.3 * exp((v + 10.0) / 70.0))
        )
        betaqi = 0.15 * 1.0 / (
            95.7 * exp(-(v + 10.0) / 10.0) + 50.0 * exp(-(v + 10.0) / 700.0)
        ) + 0.000229 / (1 + exp(-(v + 10.0) / 5.0))
        qi = alphaqi / (alphaqi + betaqi)
        tauqi = 1.0 / (alphaqi + betaqi)
        dst = dst + ddt * ((qa - dst) / tauqa)
        fst = fst + ddt * ((qi - fst) / tauqi)
        ist = gst * dst * fst * (v - eist)

        ibna = gbna * (v - ena)
        ibca = gbca * (v - eca)
        ibk = gbk * (v - ek)
        ib = ibna + ibca + ibk

        xk1inf = 1.0 / (1.0 + exp(0.070727 * (v - ek)))
        ik1 = gk1 * xk1inf * (ko / (ko + 0.228880)) * (v - ek)

        tau_dt = 1.0 / (1.068 * exp((v + 26.3) / 30.0) + 1.068 * exp(-(v + 26.3) / 30.0))
        dt_inf = 1.0 / (1.0 + exp(-(v + 26.0) / 6.0))
        dt = dt + ddt * ((dt_inf - dt) / tau_dt)
        tau_ft = 1.0 / (0.0153 * exp(-(v + 61.7) / 83.3) + 0.015 * exp((v + 61.7) / 15.38))
        ft_inf = 1.0 / (1.0 + exp((v + 61.7) / 5.6))
        ft = ft + ddt * ((ft_inf - ft) / tau_ft)
        icat = gcat * ft * dt * (v - ecat)

        ikr_act_inf = 1.0 / (1.0 + exp(-(v + 21.173694) / 9.757086))
        tau_ikr_act = 0.699821 / (
            0.003596 * exp((v) / 15.339290) + 0.000177 * exp(-(v) / 25.868423)
        )
        ikr_act = ikr_act + ddt * (ikr_act_inf - ikr_act) / tau_ikr_act
        ikr_inact_inf = 1.0 / (1.0 + exp((v + 20.758474 - 4.0) / (19.0)))
        tau_ikr_inact = 0.2 + 0.9 * 1.0 / (0.1 * exp(v / 54.645) + 0.656 * exp(v / 106.157))
        ikr_inact = ikr_inact + ddt * (ikr_inact_inf - ikr_inact) / tau_ikr_inact
        ikr = gkr * ikr_act * ikr_inact * (v - ek)

        iks_act_inf = 1.0 / (1.0 + exp(-(v - 20.876040) / 11.852723))
        tau_iks_act = 1000.0 / (
            13.097938 / (1.0 + exp(-(v - 48.910584) / 10.630272)) + exp(-(v) / 35.316539)
        )
        iks_act = iks_act + ddt * (iks_act_inf - iks_act) / tau_iks_act
        iks = gks * iks_act * iks_act * (v - eks)

        if abs(v) <= 0.001:
            alpha_dl = -28.39 * (v + 35.0) / (exp(-(v + 35.0) / 2.5) - 1.0) + 408.173
        elif abs(v + 35.0) <= 0.001:
            alpha_dl = 70.975 - 84.9 * v / (exp(-0.208 * v) - 1.0)
        elif abs(v) > 0.001 and fabs(v + 35.0) > 0.001:
            alpha_dl = -28.39 * (v + 35.0) / (exp(-(v + 35.0) / 2.5) - 1.0) - 84.9 * v / (
                exp(-0.208 * v) - 1.0
            )

        if abs(v - 5.0) <= 0.001:
            beta_dl = 28.575
        elif abs(v - 5.0) > 0.001:
            beta_dl = 11.43 * (v - 5.0) / (exp(0.4 * (v - 5.0)) - 1.0)

        tau_dl = 2000.0 / (alpha_dl + beta_dl)
        dl13_inf = 1.0 / (1 + exp(-(v + 13.5) / 6.0))
        fl13_inf = 1.0 / (1 + exp((v + 35.0) / 7.3))
        tau_fl = 7.4 + 45.77 * exp(-0.5 * (v + 28.1) * (v + 28.1) / (11 * 11))
        dl13 = dl13 + ddt * (dl13_inf - dl13) / tau_dl
        fl13 = fl13 + ddt * (fl13_inf - fl13) / tau_fl
        dl12_inf = 1.0 / (1 + exp(-(v + 3.0) / 5.0))
        fl12_inf = 1.0 / (1 + exp((v + 36.0) / 4.6))
        dl12 = dl12 + ddt * (dl12_inf - dl12) / tau_dl
        fl12 = fl12 + ddt * (fl12_inf - fl12) / tau_fl
        fca_inf = kmfca / (kmfca + casub)
        taufca = fca_inf / alpha_fca
        fca = fca + ddt * (fca_inf - fca) / taufca
        ical12 = gcal12 * fl12 * dl12 * fca * (v - ecal)
        ical13 = gcal13 * fl13 * dl13 * fca * (v - ecal)

        fna = (
            9.52e-02 * exp(-6.3e-2 * (v + 34.4)) / (1 + 1.66 * exp(-0.225 * (v + 63.7)))
        ) + 8.69e-2
        m3_inf_ttxr = 1.0 / (1.0 + exp(-(v + 45.213705) / 7.219547))
        h_inf_ttxr = 1.0 / (1.0 + exp(-(v + 62.578120) / (-6.084036)))
        m3_inf_ttxs = 1.0 / (1.0 + exp(-(v + 36.097331 - 5.0) / 5.0))
        h_inf_ttxs = 1.0 / (1.0 + exp((v + 56.0) / 3.0))
        m_inf_ttxr = pow(m3_inf_ttxr, 0.333)
        m_inf_ttxs = pow(m3_inf_ttxs, 0.333)
        tau_m = 1000.0 * (
            (
                0.6247e-03
                / (0.832 * exp(-0.335 * (v + 56.7)) + 0.627 * exp(0.082 * (v + 65.01)))
            )
            + 0.0000492
        )
        tau_h = 1000.0 * (
            (
                (3.717e-06 * exp(-0.2815 * (v + 17.11)))
                / (1 + 0.003732 * exp(-0.3426 * (v + 37.76)))
            )
            + 0.0005977
        )
        tau_j = 1000.0 * (
            (
                (0.00000003186 * exp(-0.6219 * (v + 18.8)))
                / (1 + 0.00007189 * exp(-0.6683 * (v + 34.07)))
            )
            + 0.003556
        )
        m_ttxs = m_ttxs + ddt * (m_inf_ttxs - m_ttxs) / tau_m
        h_ttxs = h_ttxs + ddt * (h_inf_ttxs - h_ttxs) / tau_h
        j_ttxs = j_ttxs + ddt * (h_inf_ttxs - j_ttxs) / tau_j
        hs = (1.0 - fna) * h_ttxs + fna * j_ttxs
        tau_mr = 1000.0 * (
            (
                0.6247e-03
                / (0.832 * exp(-0.335 * (v + 56.7)) + 0.627 * exp(0.082 * (v + 65.01)))
            )
            + 0.0000492
        )
        tau_hr = 1000.0 * (
            (
                (3.717e-06 * exp(-0.2815 * (v + 17.11)))
                / (1 + 0.003732 * exp(-0.3426 * (v + 37.76)))
            )
            + 0.0005977
        )
        tau_jr = 1000.0 * (
            (
                (0.00000003186 * exp(-0.6219 * (v + 18.8)))
                / (1 + 0.00007189 * exp(-0.6683 * (v + 34.07)))
            )
            + 0.003556
        )
        m_ttxr = m_ttxr + ddt * (m_inf_ttxr - m_ttxr) / tau_mr
        h_ttxr = h_ttxr + ddt * (h_inf_ttxr - h_ttxr) / tau_hr
        j_ttxr = j_ttxr + ddt * (h_inf_ttxr - j_ttxr) / tau_jr
        hsr = (1.0 - fna) * h_ttxr + fna * j_ttxr

        if abs(v) > 0.005:
            ina_ttxs = (
                gna_ttxs
                * m_ttxs
                * m_ttxs
                * m_ttxs
                * hs
                * nao
                * (F * F / (R * T))
                * ((exp((v - ena) * F / (R * T)) - 1.0) / (exp(v * F / (R * T)) - 1.0))
                * v
            )
        else:
            ina_ttxs = (
                gna_ttxs
                * m_ttxs
                * m_ttxs
                * m_ttxs
                * hs
                * nao
                * F
                * ((exp((v - ena) * F / (R * T)) - 1.0))
            )
        if abs(v) > 0.005:
            ina_ttxr = (
                gna_ttxr
                * m_ttxr
                * m_ttxr
                * m_ttxr
                * hsr
                * nao
                * (F * F / (R * T))
                * ((exp((v - enattxr) * F / (R * T)) - 1.0) / (exp(v * F / (R * T)) - 1.0))
                * v
            )
        else:
            ina_ttxr = (
                gna_ttxr
                * m_ttxr
                * m_ttxr
                * m_ttxr
                * hsr
                * nao
                * F
                * ((exp((v - enattxr) * F / (R * T)) - 1.0))
            )

        y_inf = 1.0 / (1.0 + exp((v + vhalf_gh) / 16.3))
        tau_y_1_2 = 1.5049 / (exp(-(v + 590.3) * 0.01094) + exp((v - 85.1) / 17.2))
        y_1_2 = y_1_2 + ddt * (y_inf - y_1_2) / tau_y_1_2
        ihk = 0.6167 * gh * y_1_2 * (v - ek)
        ihna = 0.3833 * gh * y_1_2 * (v - ena)
        ih = ihk + ihna

        q_inf = 1.0 / (1.0 + exp((v + 49.0) / 13.0))
        tau_q = (
            6.06
            + 39.102 / (0.57 * exp(-0.08 * (v + 44.0)) + 0.065 * exp(0.1 * (v + 45.93)))
        ) / 0.67
        q = q + ddt * ((q_inf - q) / tau_q)
        r_inf = 1.0 / (1.0 + exp(-(v - 19.3) / 15.0))
        tau_r = (
            2.75
            + 14.40516
            / (1.037 * exp(0.09 * (v + 30.61)) + 0.369 * exp(-0.12 * (v + 23.84)))
        ) / 0.303
        r = r + ddt * ((r_inf - r) / tau_r)
        ito = gto * q * r * (v - ek)

        isus = gsus * r * (v - ek)

        inak = (
            inakmax
            * (pow(ko, 1.2) / (pow(kmkp, 1.2) + pow(ko, 1.2)))
            * (pow(nai, 1.3) / (pow(kmnap, 1.3) + pow(nai, 1.3)))
            / (1.0 + exp(-(v - ena + 120.0) / 30.0))
        )

        di = (
            1
            + (casub / Kci) * (1 + exp(-Qci * v * FRT) + nai / Kcni)
            + (nai / K1ni) * (1 + (nai / K2ni) * (1 + nai / K3ni))
        )
        doo = (
            1
            + (cao / Kco) * (1 + exp(Qco * v * FRT))
            + (nao / K1no) * (1 + (nao / K2no) * (1 + nao / K3no))
        )
        k43 = nai / (K3ni + nai)
        k12 = (casub / Kci) * exp(-Qci * v * FRT) / di
        k14 = (nai / K1ni) * (nai / K2ni) * (1 + nai / K3ni) * exp(Qn * v * FRT / 2.0) / di
        k41 = exp(-Qn * v * FRT / 2.0)
        k34 = nao / (K3no + nao)
        k21 = (cao / Kco) * exp(Qco * v * FRT) / doo
        k23 = (
            (nao / K1no) * (nao / K2no) * (1 + nao / K3no) * exp(-Qn * v * FRT / 2.0) / doo
        )
        k32 = exp(Qn * v * FRT / 2)
        x1 = k34 * k41 * (k23 + k21) + k21 * k32 * (k43 + k41)
        x2 = k43 * k32 * (k14 + k12) + k41 * k12 * (k34 + k32)
        x3 = k43 * k14 * (k23 + k21) + k12 * k23 * (k43 + k41)
        x4 = k34 * k23 * (k14 + k12) + k21 * k14 * (k34 + k32)
        inaca = kNaCa * (k21 * x2 - k12 * x1) / (x1 + x2 + x3 + x4)
        ca_flux = (ical12 + ical13 + icat - 2.0 * inaca + ibca) / (2.0 * F)
        Jcadif = (casub - cai) / tdifca
        kcasr = maxsr - (maxsr - minsr) / (1.0 + pow(eca50sr / carel, hsrr))
        kosrca = koca / kcasr
        kisrca = kica * kcasr
        resting = resting + ddt * (
            kim * resting_inactivated
            - kisrca * casub * resting
            - kosrca * casub * casub * resting
            + kom * openg
        )
        openg = openg + ddt * (
            kosrca * casub * casub * resting
            - kom * openg
            - kisrca * casub * openg
            + kim * inactivated
        )
        inactivated = inactivated + ddt * (
            kisrca * casub * openg
            - kim * inactivated
            - kom * inactivated
            + kosrca * casub * casub * resting_inactivated
        )
        resting_inactivated = resting_inactivated + ddt * (
            kom * inactivated
            - kosrca * casub * casub * resting_inactivated
            - kim * resting_inactivated
            + kisrca * casub * resting
        )
        Jrel = ks * openg * (carel - casub)
        Jup = (
            Pup
            * (pow(cai / pumpkmf, pumphill) - pow(caup / pumpkmr, pumphill))
            / (1.0 + pow(cai / pumpkmf, pumphill) + pow(caup / pumpkmr, pumphill))
        )
        Jtr = (caup - carel) / Ttr
        dFtc = kfTC * cai * (1.0 - Ftc) - kbTC * Ftc
        dFtmc = kfTMC * cai * (1.0 - Ftmc - Ftmm) - kbTMC * Ftmc
        dFtmm = kfTMM * Mgi * (1.0 - Ftmc - Ftmm) - kbTMM * Ftmm
        dFcms = kfCM * casub * (1.0 - Fcms) - kbCM * Fcms
        dFcmi = kfCM * cai * (1.0 - Fcmi) - kbCM * Fcmi
        dFcq = kfCQ * carel * (1.0 - Fcq) - kbCQ * Fcq
        Ftc = Ftc + ddt * dFtc
        Ftmc = Ftmc + ddt * dFtmc
        Ftmm = Ftmm + ddt * dFtmm
        Fcms = Fcms + ddt * dFcms
        Fcmi = Fcmi + ddt * dFcmi
        Fcq = Fcq + ddt * dFcq
        casub = casub + ddt * ((-ca_flux + Jrel * vrel) / vsub - Jcadif - ConcCM * dFcms)
        cai = cai + ddt * (
            (Jcadif * vsub - Jup * vup) / vi
            - (ConcCM * dFcmi + ConcTC * dFtc + ConcTMC * dFtmc)
        )
        carel = carel + ddt * (Jtr - Jrel - ConcCQ * dFcq)
        caup = caup + ddt * (Jup - Jtr * vrel / vup)
        dvdtold = dvdt
        total_current = (
            ih
            + ina_ttxr
            + ina_ttxs
            + ical12
            + ical13
            + iks
            + ikr
            + ik1
            + ist
            + ib
            + icat
            + inak
            + isus
            + inaca
            + ito
        )
        dvdt = -total_current / capacitance
        vnew = v + ddt * dvdt
        ena = (R * T / F) * log(nao / nai)
        ek = (R * T / F) * log(ko / ki)
        eks = ((R * T) / F) * log((ko + 0.12 * nao) / (ki + 0.12 * nai))
        eca = (R * T / (2 * F)) * log(cao / casub)
        nai_tot = ihna + ina_ttxr + ina_ttxs + 3.0 * inak + 3.0 * inaca + ist + ibna
        ki_tot = ihk + iks + ikr + ik1 + ibk - 2.0 * inak + isus + ito
        nai = nai - ddt * (nai_tot) / (F * vi)
        ki = ki - ddt * (ki_tot) / (F * vi)

        if dvdt >= 0.0 and dvdtold < 0.0:
            min_potential[param_counter] = v
            nai_min = nai
            ki_min = ki
            tmin_potential[param_counter] = sstime
            start_output = 1

        if dvdt > dvdtmax[param_counter] and start_output > 0:
            dvdtmax[param_counter] = dvdt
            apd_start[param_counter] = sstime
            vdvdtmax[param_counter] = v

        if dvdtold > 0.0 and dvdt <= 0.0:
            max_potential[param_counter] = v
            top_slope[param_counter] = (
                max_potential[param_counter] - min_potential[param_counter]
            ) / (sstime - tmin_potential[param_counter])

        if (
            (param_counter > 0)
            and (dvdtold <= top_slope[param_counter - 1])
            and (dvdt > top_slope[param_counter - 1])
        ):
            top[param_counter] = v
            ddr[param_counter] = (v - min_potential[param_counter]) / (
                sstime - tmin_potential[param_counter]
            )

        if (
            vnew <= 0.5 * min_potential[param_counter]
            and v > 0.5 * min_potential[param_counter]
            and apd_start[param_counter] > 0.0
        ):
            apd50[param_counter] = sstime - apd_start[param_counter]

        v = vnew

        t[param_counter] = sstime
        VV[param_counter] = v

        current_trace_counter += 1

    return t, VV
