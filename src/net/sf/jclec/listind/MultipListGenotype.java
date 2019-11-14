package net.sf.jclec.listind;

import java.util.ArrayList;

/**
 * Class implementing a genotype of individual composed by a index of subpop and a list of ints.
 * 
 * @author Jose M. Moyano
 * @author Sebastian Ventura
 */
public class MultipListGenotype {
	
	/**
	 * Index of subpopulation
	 */
	public int subpop;
	
	/**
	 * Genotype of the individual as a list
	 */
	public ArrayList<Integer> genotype;

	/**
	 * Empty constructor
	 */
	public MultipListGenotype() {
		subpop = -1;
		genotype = new ArrayList<Integer>();
	}
	
	/**
	 * Constructor with parameters
	 * 
	 * @param subpop Index of subpopulation
	 * @param genotype Genotype of individual
	 */
	public MultipListGenotype(int subpop, ArrayList<Integer> genotype) {
		this.subpop = subpop;
		this.genotype = genotype;
	}
	
	/**
	 * Getter for genotype as list
	 * 
	 * @return List with genotype
	 */
	public ArrayList<Integer> getGenotype() {
		return genotype;
	}
	
	/**
	 * Getter for index of subpopulation
	 * 
	 * @return Index of subpopulation
	 */
	public int getSubpop() {
		return subpop;
	}
	
	/**
	 * Convert the MultipListGenotype into a string
	 */
	public String toString() {
		String s = "[ " + subpop + "; ";
		
		for(Integer g : genotype) {
			s += g +", ";
		}
		s = s.substring(0, s.length() - 2);
		s += "]";
		
		return s;
	}
	
	/**
	 * Clones a given object of this class.
	 * 
	 * @return Cloned object
	 */
	public MultipListGenotype clone() {
		ArrayList<Integer> newGenotype = new ArrayList<Integer>();
		for(Integer g : this.genotype) {
			newGenotype.add(g);
		}
		return new MultipListGenotype(this.subpop, newGenotype);
	}
}
