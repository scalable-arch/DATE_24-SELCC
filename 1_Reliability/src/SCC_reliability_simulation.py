# Name of the matrix files must be 'scheme_H/DE/TE/DAE/TAE_matrix_T.txt' or 'scheme_H/DE/TE/DAE/TAE_matrix.txt'.
# Required matrices for each scheme are below.
#
#   - SCC : SCC_H_matrix_T, SCC_DE_matrix_T, SCC_TE_matrix_T
#   - SEC : SEC_H_matrix
#   - SEC-DED : SEC-DED_H_matrix
#   - SEC-DAEC : SEC-DAEC_H_matrix, SEC-DAEC_DAE_matrix
#   - SEC-DAEC-TAEC : SEC-DAEC-TAEC_H_matrix, SEC-DAEC-TAEC_DAE_matrix_T, SEC-DAEC-TAEC_TAE_matrix_T

import os
import numpy as np
import random
import math
import argparse
from itertools import combinations_with_replacement

np.set_printoptions(linewidth=np.inf)


def parsing_args() :
    parser = argparse.ArgumentParser(description='Reliability simulator for 4LC/8LC PCM.')
    parser.add_argument('-w', '--width', choices=range(2, 4), default=3, type=int, help='a width of a cell')
    parser.add_argument('-s', '--scheme', choices=['SCC', 'SEC', 'SEC-DED', 'SEC-DAEC', 'SEC-DAEC-TAEC'], default='SCC', type=str, help='an ECC scheme for simulation')
    parser.add_argument('-c', '--mc', default=1000000, type=int, help='the number of iterations for Monte Carlo simulation')
    parser.add_argument('-m', '--max', default=2, type=int, help='the maximum number of erroneous cells')
    return parser.parse_args()


def make_dirs(path) :
    if not os.path.exists(path) :
        os.makedirs(path)


def gen_syndrome(codeword, H_matrix) :
    # Calculate syndrome using H-matrix
    codeword_transpose = codeword.transpose()
    syndrome = H_matrix.dot(codeword_transpose) % 2
    return syndrome


# Nerror_cell : # of erroneous cells
# Nerror_bit : # of error bits in erroneous cells
#   e.g.,) for 2 cell errors (1-bit error + 2-bit error) in 4LC,
#          gen_error(codeword_len, 2, 2, (1, 2))
def gen_error(codeword_len, cell_width, Nerror_cell, Nerror_bit) :
    error_cell_idx_arr = []
    error_pos_arr = []

    # Choose which cells to generate errors
    for Ncell in range(Nerror_cell) :
        while (True) :
            error_cell_idx = random.randrange(math.ceil(codeword_len / cell_width))
            if error_cell_idx not in error_cell_idx_arr :
                # There must not be more errors than the capacity of the selected cell
                if Nerror_bit[Ncell] * error_cell_idx + cell_width - 1 < codeword_len :
                    break
        error_cell_idx_arr.append(error_cell_idx)

    # Generate errors within the selected cells
    for Ncell in range(Nerror_cell) :
        for N in range(Nerror_bit[Ncell]) :
            while (True) :
                error_pos = random.randrange(error_cell_idx_arr[Ncell] * cell_width, error_cell_idx_arr[Ncell] * cell_width + cell_width)
                if error_pos not in error_pos_arr :
                    # Error should be within the codeword range
                    if error_pos < codeword_len :
                        break
            error_pos_arr.append(error_pos)

    # Inject errors
    codeword = np.zeros(codeword_len, dtype = int)  # no error
    for error_pos in error_pos_arr :
        codeword[error_pos] ^= 1
    err_codeword = codeword

    return err_codeword


def correction_SCC_4LC(err_codeword, syndrome, H_matrix, DE_matrix) :
    codeword_len = len(err_codeword)
    detectable = False   # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if detectable (nonzero syndrome)
    if not np.all(syndrome == 0) :
        detectable = True
        # Single error correction using H-matrix
        for error_pos in range(codeword_len) :
            if np.array_equal(H_matrix[:, error_pos], syndrome) :
                err_codeword[error_pos] ^= 1  # correct error
                correctable = True
                break
        # Single cell correction using DE-matrix
        else :
            # Double error (in single cell) correction
            for error_pos in range(codeword_len // 2) :
                if np.array_equal(DE_matrix[:, error_pos], syndrome) :
                    err_codeword[error_pos*2+0] ^= 1
                    err_codeword[error_pos*2+1] ^= 1
                    correctable = True
                    break

    return err_codeword, detectable, correctable


def correction_SCC_8LC(err_codeword, syndrome, H_matrix, DE_matrix, TE_matrix) :
    codeword_len = len(err_codeword)
    detectable = False   # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if detectable (nonzero syndrome)
    if not np.all(syndrome == 0) :
        detectable = True
        # Single error correction using H-matrix
        for error_pos in range(codeword_len) :
            if np.array_equal(H_matrix[:, error_pos], syndrome) :
                err_codeword[error_pos] ^= 1  # correct error
                correctable = True
                break
        # Single cell correction using DE-matrix and TE-matrix
        else :
            # Double error (in single cell) correction
            for error_cell_idx in range(codeword_len // 3) :
                # DE1 error correction
                if np.array_equal(DE_matrix[:, error_cell_idx*3], syndrome) :
                    err_codeword[error_cell_idx*3+0] ^= 1
                    err_codeword[error_cell_idx*3+1] ^= 1
                    correctable = True
                    break
                # DE2 error correction
                elif np.array_equal(DE_matrix[:, error_cell_idx*3+1], syndrome) :
                    err_codeword[error_cell_idx*3+1] ^= 1
                    err_codeword[error_cell_idx*3+2] ^= 1
                    correctable = True
                    break
                # DE3 error correction
                elif np.array_equal(DE_matrix[:, error_cell_idx*3+2], syndrome) :
                    err_codeword[error_cell_idx*3+0] ^= 1
                    err_codeword[error_cell_idx*3+2] ^= 1
                    correctable = True
                    break
            # Triple error (in single cell) correction
            else : 
                for error_cell_idx in range(codeword_len // 3) :  # 0, 1, ... , 23
                    if np.array_equal(TE_matrix[:, error_cell_idx], syndrome) :
                        err_codeword[error_cell_idx*3+0] ^= 1
                        err_codeword[error_cell_idx*3+1] ^= 1
                        err_codeword[error_cell_idx*3+2] ^= 1
                        correctable = True
                        break

    return err_codeword, detectable, correctable


def correction_SEC(err_codeword, syndrome, H_matrix) :
    codeword_len = len(err_codeword)
    detectable = False   # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if detectable (nonzero syndrome)
    if not np.all(syndrome == 0) :
        detectable = True
        # Single error correction using H-matrix
        for error_idx in range(codeword_len):
            if np.array_equal(H_matrix[:, error_idx], syndrome) :
                err_codeword[error_idx] ^= 1  # correct error
                correctable = True
                break

    return err_codeword, detectable, correctable


def correction_SEC_DED(err_codeword, syndrome, H_matrix) :  # same as SEC
    codeword_len = len(err_codeword)
    detectable = False   # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if detectable (nonzero syndrome)
    if not np.all(syndrome == 0) :
        detectable = True
        # Single error correction using H-matrix
        for error_idx in range(codeword_len):
            if np.array_equal(H_matrix[:, error_idx], syndrome) :
                err_codeword[error_idx] ^= 1  # correct error
                correctable = True
                break

    return err_codeword, detectable, correctable


def correction_SEC_DAEC(err_codeword, syndrome, H_matrix, DAE_matrix) :
    codeword_len = len(err_codeword)
    detectable = False   # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if detectable (nonzero syndrome)
    if not np.all(syndrome == 0) :
        detectable = True
        # Single error correction using H-matrix
        for error_idx in range(codeword_len):
            if np.array_equal(H_matrix[:, error_idx], syndrome) :
                err_codeword[error_idx] ^= 1  # correct error
                correctable = True
                break
        # Double adjacent error correction using DAE-matrix
        else :
            for error_idx in range(codeword_len - 1):
                if np.array_equal(DAE_matrix[:, error_idx], syndrome) :
                    err_codeword[error_idx] ^= 1  # correct error
                    err_codeword[error_idx + 1] ^= 1
                    correctable = True
                    break

    return err_codeword, detectable, correctable


def correction_SEC_DAEC_TAEC(err_codeword, syndrome, H_matrix, DAE_matrix, TAE_matrix) :
    codeword_len = len(err_codeword)
    detectable = False   # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if detectable (nonzero syndrome)
    if not np.all(syndrome == 0) :
        detectable = True
        # Single error correction using H-matrix
        for error_idx in range(codeword_len):
            if np.array_equal(H_matrix[:, error_idx], syndrome) :
                err_codeword[error_idx] ^= 1  # correct error
                correctable = True
                break
        else :
            # Double adjacent error correction using DAE-matrix
            for error_idx in range(codeword_len - 1) :
                if np.array_equal(DAE_matrix[:, error_idx], syndrome) :
                    err_codeword[error_idx] ^= 1
                    err_codeword[error_idx + 1] ^= 1
                    correctable = True
                    break
            # Triple adjacent error correction using TAE-matrix
            else :
                for error_idx in range(codeword_len - 2) :
                    if np.array_equal(TAE_matrix[:, error_idx], syndrome) :
                        err_codeword[error_idx] ^= 1  # correct error
                        err_codeword[error_idx + 1] ^= 1
                        err_codeword[error_idx + 2] ^= 1
                        correctable = True
                        break

    return err_codeword, detectable, correctable


def verify(corr_codeword, detectable, correctable, CE_cnt, DUE_cnt, SDC_cnt) :
    # Verify if the error correction has successfully finished
    if not detectable :
        SDC_cnt += 1          # error but zero syndrome
    else :
        if not correctable :
            DUE_cnt += 1      # detected but not correctable
        else :
            if not np.all(corr_codeword == 0) :
                SDC_cnt += 1  # detected but miscorrected
            else :
                CE_cnt += 1   # detected and correctly corrected

    return CE_cnt, DUE_cnt, SDC_cnt


def main() :
    # Argument parsing
    args = parsing_args()
    cell_width = args.width
    scheme = args.scheme
    mc_sim_cnt = args.mc
    max_Nerror_cell = args.max
        
    # Make an output file
    path = "../output/" + str(2 ** cell_width) + "LC/"
    make_dirs(path)
    output_file = open(path + scheme + "_result.txt", 'w')

    # Setting the matrices from source files
    if scheme == "SCC" :
        if cell_width == 2 :
            H_matrix = np.loadtxt("SCC/4LC/SCC_H_matrix_T.txt", dtype = "int").transpose()
            DE_matrix = np.loadtxt("SCC/4LC/SCC_DE_matrix_T.txt", dtype = "int").transpose()
        elif cell_width == 3 :
            H_matrix = np.loadtxt("SCC/8LC/SCC_H_matrix_T.txt", dtype = "int").transpose()
            DE_matrix = np.loadtxt("SCC/8LC/SCC_DE_matrix_T.txt", dtype = "int").transpose()
            TE_matrix = np.loadtxt("SCC/8LC/SCC_TE_matrix_T.txt", dtype = "int").transpose()
    elif scheme == "SEC" :  # only for 4LC
        H_matrix = np.loadtxt("SEC/SEC_H_matrix.txt", dtype = "int")
    elif scheme == "SEC-DED" :
        H_matrix = np.loadtxt("SEC-DED/SEC-DED_H_matrix.txt", dtype = "int")
    elif scheme == "SEC-DAEC" :
        H_matrix = np.loadtxt("SEC-DAEC/SEC-DAEC_H_matrix.txt", dtype = "int")
        DAE_matrix = np.loadtxt("SEC-DAEC/SEC-DAEC_DAE_matrix_T.txt", dtype = "int").transpose()
    elif scheme == "SEC-DAEC-TAEC" :  # only for 8LC
        H_matrix = np.loadtxt("SEC-DAEC-TAEC/SEC-DAEC-TAEC_H_matrix.txt", dtype = "int")
        DAE_matrix = np.loadtxt("SEC-DAEC-TAEC/SEC-DAEC-TAEC_DAE_matrix_T.txt", dtype = "int").transpose()
        TAE_matrix = np.loadtxt("SEC-DAEC-TAEC/SEC-DAEC-TAEC_TAE_matrix_T.txt", dtype = "int").transpose()

    # Print the code configuration
    redundancy_len = len(H_matrix)
    codeword_len = len(H_matrix[0])
    data_len = codeword_len - redundancy_len
    print("-----------------------------------------------------------")
    print("  ({0},{1}) {2} code for {3}LC memory".format(codeword_len, data_len, scheme, 2 ** cell_width))
    print("-----------------------------------------------------------")

    output_file.write("-----------------------------------------------------------\n")
    output_file.write("  (%d,%d) %s code for %dLC memory\n" % (codeword_len, data_len, scheme, 2 ** cell_width))
    output_file.write("-----------------------------------------------------------\n")

    # Now the evaluation starts.
    # Evaluated error scenarios are below.
    #
    #   1. 1-cell error
    #      1) 1-bit error
    #      2) 2-bit error
    #      3) 3-bit error (only for 8LC)
    #
    #   2. 2-cell error
    #      1) 1-bit error + 1-bit error
    #      2) 1-bit error + 2-bit error
    #      3) 1-bit error + 3-bit error (only for 8LC)
    #      4) 2-bit error + 2-bit error
    #      5) 2-bit error + 3-bit error (only for 8LC)
    #      6) 3-bit error + 3-bit error (only for 8LC)

    Nerror_bit_arr = []
    for N in range(1, cell_width + 1) :
        Nerror_bit_arr.append(N)

    # Evaulate all the cases
    for Nerror_cell in range(1, max_Nerror_cell + 1) :
        for Nerror_bit in combinations_with_replacement(Nerror_bit_arr, Nerror_cell):
            # Monte-Carlo simulation
            CE_cnt = 0
            DUE_cnt = 0
            SDC_cnt = 0
        
            for Nrun in range (0, mc_sim_cnt) :
                # Error injection
                err_codeword = gen_error(codeword_len, cell_width, Nerror_cell, Nerror_bit)
                
                # Decoding
                syndrome = gen_syndrome(err_codeword, H_matrix)
                if scheme == "SCC" :
                    if cell_width == 2 :
                        corr_codeword, detectable, correctable = correction_SCC_4LC(err_codeword, syndrome, H_matrix, DE_matrix)
                    elif cell_width == 3 :
                        corr_codeword, detectable, correctable = correction_SCC_8LC(err_codeword, syndrome, H_matrix, DE_matrix, TE_matrix)
                elif scheme == "SEC" :
                    corr_codeword, detectable, correctable = correction_SEC(err_codeword, syndrome, H_matrix)
                elif scheme == "SEC-DED" :
                    corr_codeword, detectable, correctable = correction_SEC_DED(err_codeword, syndrome, H_matrix)
                elif scheme == "SEC-DAEC" :
                    corr_codeword, detectable, correctable = correction_SEC_DAEC(err_codeword, syndrome, H_matrix, DAE_matrix)
                elif scheme == "SEC-DAEC-TAEC" :
                    corr_codeword, detectable, correctable = correction_SEC_DAEC_TAEC(err_codeword, syndrome, H_matrix, DAE_matrix, TAE_matrix)

                CE_cnt, DUE_cnt, SDC_cnt = verify(corr_codeword, detectable, correctable, CE_cnt, DUE_cnt, SDC_cnt)

            # Print the results
            print("-----------------------------------------------------------")
            print("   * {0}-cell error : ".format(Nerror_cell), end='')
            for N in range(Nerror_cell) :
                print("{0}-bit error".format(Nerror_bit[N]), end='')
                if N == Nerror_cell - 1 :  # the last iter
                    print("")
                else :
                    print(" + ", end='')
            print("     - CE  : {0} / {1} ({2}%)".format(CE_cnt, mc_sim_cnt, CE_cnt * 100 / mc_sim_cnt))
            print("     - DUE : {0} / {1} ({2}%)".format(DUE_cnt, mc_sim_cnt, DUE_cnt * 100 / mc_sim_cnt))
            print("     - SDC : {0} / {1} ({2}%)".format(SDC_cnt, mc_sim_cnt, SDC_cnt * 100 / mc_sim_cnt))
            print("-----------------------------------------------------------")

            output_file.write("-----------------------------------------------------------\n")
            output_file.write("   * %d-cell error : " % (Nerror_cell))
            for N in range(Nerror_cell) :
                output_file.write("%d-bit error" % (Nerror_bit[N]))
                if N == Nerror_cell - 1 :  # the last iter
                    output_file.write("\n")
                else :
                    output_file.write(" + ")
            output_file.write("     - CE  : %d / %d (%.4f%%)\n" % (CE_cnt, mc_sim_cnt, CE_cnt * 100 / mc_sim_cnt))
            output_file.write("     - DUE  : %d / %d (%.4f%%)\n" % (DUE_cnt, mc_sim_cnt, DUE_cnt * 100 / mc_sim_cnt))
            output_file.write("     - SDC  : %d / %d (%.4f%%)\n" % (SDC_cnt, mc_sim_cnt, SDC_cnt * 100 / mc_sim_cnt))
            output_file.write("-----------------------------------------------------------\n")

    output_file.close()

if __name__ == "__main__":
    main()