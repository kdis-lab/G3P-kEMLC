package g3pkemlc.utils;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;

import g3pkemlc.utils.Utils.KMode;
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
	public int minK;
	
	/**
	 * Max size of the k-labelsets
	 */
	public int maxK;
	
	/**
	 * Number of labels in the dataset
	 */
	public int nLabels;
	
	/**
	 * Number of k-labelsets created
	 */
	public int nLabelsets;
	
	/**
	 * Number of klabelsets of each size
	 */
	public int [] kSizes;
	
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
	 * Mode of k selection for each k-labelset.
	 * By default it is gaussian.
	 */
	KMode kMode;
	
	/**
	 * Empty constructor
	 */
	public KLabelsetGenerator() {
		this.minK = 0;
		this.maxK = 0;
		this.nLabels = 0;
		this.nLabelsets = 0;
		this.phiBiased = false;
		this.klabelsets = null;
	}
	
	/**
	 * Constructor
	 * 
	 * @param minK Minimum allowed size of k-labelsets
	 * @param maxK Maximum allowed size of k-labelsets
	 * @param nLabels Number of labels in the dataset
	 */
	public KLabelsetGenerator(int minK, int maxK, int nLabels, KMode kMode) {
		this.minK = minK;
		this.maxK = maxK;
		this.nLabels = nLabels;
		this.kMode = kMode;
		this.nLabelsets = 0;
		this.phiBiased = false;
		this.klabelsets = new ArrayList<KLabelset>();
		this.kSizes = new int[maxK - minK + 1];
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
	 * Generate k-labelset biased by phi correlation among labels
	 * Also considers the appearances in the initial pool so far
	 * 
	 * @param k Size of the labelset to generate
	 * @return Phi-biased generated k-labelset
	 */
	private KLabelset phiBasedKLabelset(int k, int[] appearances) {
		double[] weights = new double[nLabels];
		
		//Labels selected for the k-labelset
		ArrayList<Integer> klabelset = new ArrayList<Integer>(k);
		
		//Non-selected labels. Include all labels at the beginning
		ArrayList<Integer> inactive = new ArrayList<Integer>(nLabels);
		for(int i=0; i<nLabels; i++) {
			inactive.add(i);
		}
		
		//Select first label randomly
		int r = randgen.choose(0, nLabels);
		klabelset.add(r);
		inactive.remove((Integer)r);
		
		//Select the rest of labels
		do {
			//Reset weights
			Arrays.fill(weights, 0);
			
			//Recalculate weights for each inactive label
			for(Integer inLabel : inactive) {
				//Fill positions with initial weight to avoid 0 weights
				weights[inLabel] = 1E-3;
				
				//Add phi value (absolute value) with each activel abel
				for(Integer acLabel : klabelset) {
					weights[inLabel] += Math.abs(phi[inLabel][acLabel]);
				}
				
				//Multiply weights given the appearances in the initial pool
				//	So labels with very low appearance have more chance to be selected
				//	But still considering the relationship
				//It follows an exponential function, so given its appearances is multiplied by
				//	0:  2.72
				//	1:  1.65
				//	2:  1.4
				//	...
				//	5:  1.18
				//	10: 1.10
				//	20: 1.05
				weights[inLabel] *= Math.exp(1 / (1+appearances[inLabel]));
			}
			
			do {
				//Select a random label based on calculated weights
				r = utils.selectRandomlyWithWeights(weights);
			}while(klabelset.contains(r));

			klabelset.add(r);
			inactive.remove((Integer)r);
		}while(klabelset.size() < k);
		
		//Sort the klabelset
		Collections.sort(klabelset);
				
		return new KLabelset(k, this.nLabels, klabelset);
	}
	
	/**
	 * Generate an array of k-labelsets
	 * The size of each k-labelset is selected in the range [minK, maxK] (uniformly or gaussian-based)
	 * k-labelsets are created until v average votes for each label are reached
	 * 
	 * @param v Average number of votes for each label
	 * @return Array of KLabelsets
	 */
	public ArrayList<KLabelset> generateKLabelsets(int v){		
		//Clear k-labelsets array
		this.klabelsets = new ArrayList<KLabelset>(nLabelsets);
		
		
		//Current number of votes in total
		int currentVotes = 0;
		
		//Expected number of votes in the array of k-labelset
		int expectedVotes = v*nLabels;
		
		//Appearances of each label in the initial pool so far
		int [] appearances = new int[nLabels];
		
		//Next k-labelset to include
		KLabelset nextKLabelset;
		
		//Value of k for the current k-labelset
		int currentK = 0;
		
		//Sigma for selecting based on gaussian
		double sigma, r;

		do {
			if(kMode == KMode.uniform) {
				currentK = randgen.choose(minK, maxK+1);
			}
			else if(kMode == KMode.gaussian){
				//98.7% of random numbers will be in the range [minK, maxK]
				sigma = (maxK-minK)/2.5;
				
				//Get random value based on a gaussian with mean=0
				r = randgen.gaussian(sigma);
				
				//Calculate absolute value and add the minK value (so it is as a gaussian with mean=minK and only right part
				currentK = (int) Math.floor(Math.abs(r)) + minK;
				
				//Fix if the selected k is lower than minK or greater than maxK
				if(currentK < minK) {
					currentK = minK;
				}
				else if (currentK > maxK){
					currentK = maxK;
				}
			}
			else {
				System.out.println("Error. Invalid kMode.");
				System.exit(0);
			}

			//Select k-labelset (phi-biased or randomly)
			if(phiBiased) {
				nextKLabelset = phiBasedKLabelset(currentK, appearances);
			}
			else {
				//Add a randomly generated k-labelset if it did not exist
				nextKLabelset = randomKLabelset(currentK);
			}
			
			//Add if it is new
			if(! klabelsets.contains(nextKLabelset)) {
				klabelsets.add(nextKLabelset);
				this.kSizes[currentK - minK]++;
				currentVotes += nextKLabelset.klabelset.size();
				for(int l : nextKLabelset.klabelset) {
					appearances[l]++;
				}
			}
		//Add k-labelsets until desired size
		}while(currentVotes <= (expectedVotes - maxK/2));
		
		nLabelsets = klabelsets.size();
		
		int r1, r2;
		for(int l=0; l<nLabels; l++) {
			//If any label does not appear, replace it by other
			if(appearances[l] < 1) {
				int selected=0;
				do {
					//Select a random labelset
					r1 = randgen.choose(0, nLabelsets);
					
					//Select a random label in the labelset
					r2 = randgen.choose(0, klabelsets.get(r1).klabelset.size());
					
					//Selected label
					selected = klabelsets.get(r1).klabelset.get(r2);
					
					if(appearances[selected] > 1) {
						klabelsets.get(r1).klabelset.remove(r2);
						appearances[selected]--;
						klabelsets.get(r1).klabelset.add(l);
						appearances[l]++;
					}
				}while(appearances[selected] < 1);
				
			}
		}
		
		System.out.println("Votes in pool: " + Arrays.toString(appearances));
		
		return this.klabelsets;
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
		s +=  "nL: " + nLabels + "\n";
		
		s += "k sizes in pool: ";
		for(int i=0; i<kSizes.length; i++) {
			s += kSizes[i] + ", ";
		}
		s += "\n";
		
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
