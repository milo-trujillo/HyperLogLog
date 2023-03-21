#!/usr/bin/env python3
import hashlib
import numpy as np

# MD5 digest as text bitstring
def hashBits(term):
	bitstring = ""
	digest = hashlib.md5(term.encode("utf-8")).digest()
	for byte in digest:
		bits = bin(byte)[2:].rjust(8, "0")
		bitstring += bits
	return bitstring

# Returns number of leading 0s in a bitstring
def leadingZeros(bitstring):
	leading = 0
	for bit in bitstring:
		if bit == "0":
			leading += 1
		else:
			return leading
	return leading

# Expects list of text strings, returns estimated unique elements based
# on leading zeros of hashes
def probabilisticCount(terms):
	leading_zeros = 0
	for term in terms:
		zeros = leadingZeros(hashBits(term))
		leading_zeros = max(leading_zeros, zeros)
	return 2 ** leading_zeros

# Calculates ten hashes per string, an expected number of unique values
# based on each hash function, then returns either median or mean of those
# ten estimates
def probabilisticCountAveraged(terms, avg="median"):
	answers = []
	for salt in ["", "0", "1", "2", "3", "4", "5", "6", "7", "8"]:
		leading_zeros = 0
		for term in terms:
			zeros = leadingZeros(hashBits(term+salt))
			leading_zeros = max(leading_zeros, zeros)
		answers.append(2**leading_zeros)
	if( avg == "median" ):
		return np.median(answers)
	elif( avg == "mean" ):
		return np.mean(answers)
	else:
		raise Exception("Invalid type of average")

# Takes list of token strings, hashes and bins them, returns estimated unique
# tokens. Creates 2**binbits bins, so 16 bins for four binbits
def hyperLogLog(terms, binbits=4):
	bincount = 2**binbits
	bins = [0]*bincount
	for term in terms:
		bitstring = hashBits(term)
		binIndex = int(bitstring[0:binbits], 2)
		zeros = leadingZeros(bitstring[binbits:])
		bins[binIndex] = max(bins[binIndex], zeros)
	# Returns magic number to boost estimates based on deflation from bincount	
	def binCompensator(bins):
		try:
			return {16: 0.673, 32: 0.697, 64: 0.709}[bins]
		except KeyError:
			if( bins >= 128 ):
				return 0.7212 / (1 + 1.079/bins)
			raise ValueError("Invalid number of bins -- minimum 16")
	def harmonicAverage(bins):
		return sum(map(lambda b: 2**(-1*(1+b)), bins)) ** -1
	return binCompensator(bincount) * (bincount**2) * harmonicAverage(bins)
