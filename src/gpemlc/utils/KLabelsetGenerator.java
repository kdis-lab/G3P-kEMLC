package gpemlc.utils;

import java.util.ArrayList;
import java.util.Collections;

import net.sf.jclec.util.random.IRandGen;

public class KLabelsetGenerator {

	/**
	 * Size of the k-labelsets
	 */
	int k;
	
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
		this.k = 0;
		this.nLabels = 0;
		this.nLabelsets = 0;
		this.freqBias = false;
		this.freq = null;
		this.klabelsets = null;
	}
	
	/**
	 * Constructor
	 * 
	 * @param k Size of the k-labelsets
	 * @param nLabels Number of labels in the dataset
	 */
	public KLabelsetGenerator(int k, int nLabels) {
		this.k = k;
		this.nLabels = nLabels;
		this.nLabelsets = 0;
		this.freqBias = false;
		this.freq = null;
		this.klabelsets = new ArrayList<KLabelset>();
	}
	
	/**
	 * Constructor
	 * 
	 * @param k Size of the k-labelsets
	 * @param nLabels Number of labels in the dataset
	 * @param nLabelsets Number of labelsets to generate
	 */
	public KLabelsetGenerator(int k, int nLabels, int nLabelsets) {
		this.k = k;
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
	 * @return Randomly generated k-labelset
	 */
	private KLabelset randomKLabelset() {
		ArrayList<Integer> klabelset = new ArrayList<Integer>(k);
		
		int r;
		do {
			r = randgen.choose(0, nLabels);
			if(! klabelset.contains(r)) {
				klabelset.add(r);
			}
		}while(klabelset.size() < k);
		
		Collections.sort(klabelset);
		
		return new KLabelset(this.k, this.nLabels, klabelset);
	}
	
	public ArrayList<KLabelset> generateKLabelsets(int nLabelsets){
		//Clear k-labelsets array
		this.klabelsets = new ArrayList<KLabelset>(nLabelsets);
		KLabelset nextKLabelset;
		
		//Generate random k-labelsets
		if(! freqBias) {
			do {
				//Add a randomly generated k-labelset if it did not exist
				nextKLabelset = randomKLabelset();
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
	
	public void printKLabelsets() {
		System.out.println("k: " + k + "; nL: " + nLabels + "; --> " + klabelsets.toString());
	}
}
