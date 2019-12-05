package gpemlc.utils;

import java.util.ArrayList;

public class KLabelset {

	/**
	 * Size of the k-labelset
	 */
	int k;
	
	/**
	 * Number of labels in the dataset
	 */
	int nLabels;
	
	/**
	 * k-labelset
	 */
	ArrayList<Integer> klabelset;
	
	/**
	 * Empty constructor
	 */
	public KLabelset() {
		k = 0;
		nLabels = 0;
		klabelset = null;
	}
	
	/**
	 * Constructor
	 * 
	 * @param k Size of the k-labelset
	 * @param nLabels Number of labels in the dataset
	 */
	public KLabelset(int k, int nLabels) {
		this.k = k;
		this.nLabels = nLabels;
		this.klabelset = new ArrayList<Integer>(k);
	}
	
	/**
	 * Constructor 
	 * 
	 * @param k Size of the k-labelset
	 * @param nLabels Number of labels in the dataset
	 * @param klabelset k-labelset
	 */
	public KLabelset(int k, int nLabels, ArrayList<Integer> klabelset) {
		this.k = k;
		this.nLabels = nLabels;
		this.klabelset = new ArrayList<Integer>(klabelset);
	}
	
	/**
	 * Constructor 
	 * 
	 * @param nLabels Number of labels in the dataset
	 * @param klabelset k-labelset
	 */
	public KLabelset(int nLabels, ArrayList<Integer> klabelset) {
		this.k = klabelset.size();
		this.nLabels = nLabels;
		this.klabelset = new ArrayList<Integer>(klabelset);
	}
	
	/**
	 * Getter for KLabelset
	 * 
	 * @return k-labelset
	 */
	public ArrayList<Integer> getKlabelset() {
		return klabelset;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj.getClass() == this.getClass()) { //Check class
			KLabelset other = (KLabelset) obj;
			
			if(other.k == this.k && other.nLabels == this.nLabels) { //Check k and nLabels
				for(int i=0; i<this.k; i++){
					if(! other.klabelset.contains(this.klabelset.get(i))) { //If any label is not included in the labelset, return false
						return false;
					}
				}
				return true; //If all labels are included, return true
			}
		}
		
		return false;
	}

	@Override
	public String toString() {
		return klabelset.toString();
	}
}
