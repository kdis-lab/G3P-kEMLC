package g3pkemlc.utils;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
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
	 * True if the generation of the k-labelsets is biased by the relationship among labels.
	 * If false, they are randomly generated.
	 */
	boolean phiBiased;
	
	/**
	 * Phi matrix of correlation among labels
	 */
	double[][] phi;
	
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
	 * Utils
	 */
	Utils utils = new Utils();
	
	/**
	 * Empty constructor
	 */
	public KLabelsetGenerator() {
		this.minK = 0;
		this.maxK = 0;
		this.nLabels = 0;
		this.nLabelsets = 0;
		this.phiBiased = false;
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
		this.phiBiased = false;
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
		this.phiBiased = false;
		this.freq = null;
		this.klabelsets = new ArrayList<KLabelset>(nLabelsets);
	}
	
	/**
	 * Setter for phiBiased
	 * 
	 * @param phiBiased true if the k-labelset generation is biased by the relationship among labels
	 */
	public void setPhiBiased(boolean phiBiased, double[][] phi) {
		this.phiBiased = phiBiased;
		this.phi = phi;
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
	 * Generate k-labelset biased by phi correlation among labels
	 * 
	 * @param k Size of the labelset to generate
	 * @return Phi-biased generated k-labelset
	 */
	private KLabelset phiBasedKLabelset(int k, int[] appearances) {
		double[] weights = new double[nLabels];
		
		ArrayList<Integer> klabelset = new ArrayList<Integer>(k);
		ArrayList<Integer> inactive = new ArrayList<Integer>(nLabels);
		for(int i=0; i<nLabels; i++) {
			inactive.add(i);
		}
		
		//Select first label randomly
		int r = randgen.choose(0, nLabels);
		klabelset.add(r);
		inactive.remove((Integer)r);
		
		//Add labels until desired size
		Arrays.fill(weights, 0.0001); //
		do {
			//Recalculate weights
			for(Integer inLabel : inactive) {
				for(Integer acLabel : klabelset) {
					weights[inLabel] += Math.abs(phi[inLabel][acLabel]);
				}
			}
			
			float log = 0;
			for(int i=0; i<nLabels; i++) {
				log = (float) Math.log10(appearances[i]+1);
				if(log < 1) {
					weights[i] += (1-log);
				}
			}
			
			do {
				r = utils.selectRandomlyWithWeights(weights);
			}while(klabelset.contains(r));
			
			
			
			klabelset.add(r);
			inactive.remove((Integer)r);
		}while(klabelset.size() < k);
		
		Collections.sort(klabelset);
				
		return new KLabelset(k, this.nLabels, klabelset);
	}
	
	/**
	 * Generate a phi-based k-labelset being k in the range [minK, maxK]
	 * 
	 * @param minK Minimum allowed value for k
	 * @param maxK Maximum allowed value for k
	 * @return Phi-based generated k-labelset
	 */
	private KLabelset phiBasedKLabelset(int minK, int maxK, int [] appearances) {
		return phiBasedKLabelset(randgen.choose(minK, maxK+1), appearances);
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
		
		int [] appearances = new int[nLabels];
		
		do {
			if(! phiBiased) {
				//Add a randomly generated k-labelset if it did not exist
				nextKLabelset = randomKLabelset(this.minK, this.maxK);
			}
			else {
				nextKLabelset = phiBasedKLabelset(this.minK, this.maxK, appearances);
			}
			
			if(! klabelsets.contains(nextKLabelset)) {
				klabelsets.add(nextKLabelset);
				for(int l : nextKLabelset.klabelset) {
					appearances[l]++;
				}
			}
		}while(klabelsets.size() < nLabelsets);
		
		for(int l=0; l<nLabels; l++) {
			if(appearances[l] < 1) {
				int selected=0;
				do {
					int r = randgen.choose(0, nLabelsets);
					int r2 = randgen.choose(0, klabelsets.get(r).klabelset.size());
					selected = klabelsets.get(r).klabelset.get(r2);
					
					if(appearances[selected] > 1) {
						klabelsets.get(r).klabelset.remove(r2);
						appearances[selected]--;
						klabelsets.get(r).klabelset.add(l);
						appearances[l]++;
					}
				}while(appearances[selected] <= 1);
				
			}
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
