package gpemlc.utils;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Locale;

import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the generator of k-labelsets
 * 
 * @author Jose M. Moyano
 *
 */
public class KLabelsetGenerator {

	/**
	 * Min size of the k-labelsets
	 */
	int minK;
	
	/**
	 * Max size of the k-labelsets
	 */
	int maxK;
	
	/**
	 * Number of labels in the dataset
	 */
	int nLabels;
	
	/**
	 * Number of k-labelsets to create
	 */
	int nLabelsets;
	
	/**
	 * True if the generation of the k-labelsets is biased by the frequency of the labels.
	 * If false, they are randomly generated.
	 */
	boolean freqBias;
	
	/**
	 * Frequency of labels in the dataset. Used if freqBias is true.
	 */
	double[] freq;
	
	/**
	 * Set of selected k-labelsets
	 */
	ArrayList<KLabelset> klabelsets;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Empty constructor
	 */
	public KLabelsetGenerator() {
		this.minK = 0;
		this.maxK = 0;
		this.nLabels = 0;
		this.nLabelsets = 0;
		this.freqBias = false;
		this.freq = null;
		this.klabelsets = null;
	}
	
	/**
	 * Constructor
	 * 
	 * @param minK Minimum allowed size of k-labelsets
	 * @param maxK Maximum allowed size of k-labelsets
	 * @param nLabels Number of labels in the dataset
	 */
	public KLabelsetGenerator(int minK, int maxK, int nLabels) {
		this.minK = minK;
		this.maxK = maxK;
		this.nLabels = nLabels;
		this.nLabelsets = 0;
		this.freqBias = false;
		this.freq = null;
		this.klabelsets = new ArrayList<KLabelset>();
	}
	
	/**
	 * Constructor
	 * 
	 * @param minK Minimum allowed size of k-labelsets
	 * @param maxK Maximum allowed size of k-labelsets
	 * @param nLabels Number of labels in the dataset
	 * @param nLabelsets Number of labelsets to generate
	 */
	public KLabelsetGenerator(int minK, int maxK, int nLabels, int nLabelsets) {
		this.minK = minK;
		this.maxK = maxK;
		this.nLabels = nLabels;
		this.nLabelsets = nLabelsets;
		this.freqBias = false;
		this.freq = null;
		this.klabelsets = new ArrayList<KLabelset>(nLabelsets);
	}
	
	/**
	 * Setter for freqBias
	 * 
	 * @param freqBias true if the k-labelset generation is biased by the frequency of labels
	 */
	public void setFreqBias(boolean freqBias) {
		this.freqBias = freqBias;
	}
	
	/**
	 * Setter for the frequency
	 * 
	 * @param freq Frequency of each label in the dataset
	 */
	public void setFreq(double[] freq) {
		this.freq = freq;
	}
	
	/**
	 * Setter for randgen
	 * 
	 * @param randgen Random numbers generator
	 */
	public void setRandgen(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	/**
	 * Generate random k-labelset
	 * 
	 * @param k Size of the labelset to generate
	 * @return Randomly generated k-labelset
	 */
	private KLabelset randomKLabelset(int k) {
		ArrayList<Integer> klabelset = new ArrayList<Integer>(k);
		
		int r;
		do {
			r = randgen.choose(0, nLabels);
			if(! klabelset.contains(r)) {
				klabelset.add(r);
			}
		}while(klabelset.size() < k);
		
		Collections.sort(klabelset);
		
		return new KLabelset(k, this.nLabels, klabelset);
	}
	
	/**
	 * Generate a random k-labelset being k in the range [minK, maxK]
	 * 
	 * @param minK Minimum allowed value for k
	 * @param maxK Maximum allowed value for k
	 * @return Randomly generated k-labelset
	 */
	private KLabelset randomKLabelset(int minK, int maxK) {
		return randomKLabelset(randgen.choose(minK, maxK+1));
	}
	
	/**
	 * Generate an array of nLabelsets k-labelsets
	 * The size of each k-labelset is randomly selected in the range [minK, maxK]
	 * 
	 * @param nLabelsets Number of KLabelsets to generate
	 * @return Array of KLabelsets
	 */
	public ArrayList<KLabelset> generateKLabelsets(int nLabelsets){
		//Clear k-labelsets array
		this.klabelsets = new ArrayList<KLabelset>(nLabelsets);
		KLabelset nextKLabelset;
		
		//Generate random k-labelsets
		if(! freqBias) {
			do {
				//Add a randomly generated k-labelset if it did not exist
				nextKLabelset = randomKLabelset(this.minK, this.maxK);
				
				if(! klabelsets.contains(nextKLabelset)) {
					klabelsets.add(nextKLabelset);
				}
			}while(klabelsets.size() < nLabelsets);
		}
		else {
			
		}
		
		return this.klabelsets;
	}
	
	/**
	 * Generate k-labelsets
	 * 
	 * @return List of k-labelsets
	 */
	public ArrayList<KLabelset> generateKLabelsets(){
		if(this.nLabelsets > 0) {
			return generateKLabelsets(this.nLabelsets);
		}
		
		return null;
	}
	
	/**
	 * Print the KLabelsets object
	 */
	public void printKLabelsets() {
		String s = new String();
		
		//Add k range
		s += "k: [" + minK + ", " + maxK + "]; ";
		
		//Add avgK
		DecimalFormat df = (DecimalFormat)DecimalFormat.getNumberInstance(Locale.ENGLISH);
		s += "avgK: " + df.format(avgSizeKLabelsets()) + "; ";
		
		//Add nLabels
		s +=  "nL: " + nLabels + "; ";
		
		//Add k-labelsets
		s += "--> " + klabelsets.toString();
		
		System.out.println(s);
	}
	
	/**
	 * Compute the average size of the k-labelsets in the array
	 * 
	 * @return Average size of the k-labelsets
	 */
	public double avgSizeKLabelsets() {
		double sum = 0.0;
		
		for(KLabelset kl : this.klabelsets) {
			sum += kl.k;
		}
		
		return sum/this.klabelsets.size();
	}
}
