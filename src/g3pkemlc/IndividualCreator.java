package g3pkemlc;

import java.util.Locale;

import g3pkemlc.utils.Utils;
import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the individual creator for G3P-kEMLC individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class IndividualCreator {

	/**
	 * Utils object for working with GP individuals
	 */
	Utils utils = new Utils();
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Standard deviation parameter for gaussian to calculate thresholds
	 */
	double gaussianStdv;
	
	/**
	 * Constructor
	 * 
	 * @param randgen Random numbers generator
	 */
	public IndividualCreator(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	/**
	 * Setter for randgen
	 * @param rangen
	 */
	public void setRangen(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	/**
	 * Create an individual of given maximum depth
	 * 
	 * @param nMax Max value for the leaves
	 * @param maxDepth Maximum depth of the tree
	 * @return Individual as String
	 */
	public String create(int nMax, int maxDepth, int minChildren, int maxChildren, double gaussianStdv) {
		//Set stdv for gaussian thresholds
		this.gaussianStdv = gaussianStdv;
		
		//The individual at the beginning is the node 'S' and the end of individual ';'
		String ind = "S;";
		
		//Current position creating the individual
		int pos = 0;
		
		//Depth of the current node
		int currDepth = 0;
				
		do {
			switch (ind.charAt(pos)) {
			//If initial 'S' node, create a random number of child nodes 'C'.
				//Random in [minChildren, maxChild] range
			case 'S':
				ind = replace(ind, pos, childRandomSize(minChildren, maxChildren));
				break;
			
			//If a node 'C' is found, it is replaced by one of:
			//	- Random number of child in [Children, maxChild] range
			//	- Random leaf
			//If the current depth is equal than the max allowed depth, a random leaf is automatically included
			case 'C':
				if(currDepth < maxDepth) {
					if(randgen.coin()) {
						ind = replace(ind, pos, childRandomSize(minChildren, maxChildren));
					}
					else {
						//Replace current 'C' by random leaf in [0, nMax) range
						ind = replace(ind, pos, String.valueOf(randgen.choose(0, nMax)));
					}
				}
				else {
					//Replace current 'C' by random leaf in [0, nMax) range
					ind = replace(ind, pos, String.valueOf(randgen.choose(0, nMax)));
				}				
				break;
				
			//If a start parenthesis is found, a new node starts so the current depth is incremented
			case '(':
				currDepth++;
				pos++;
				break;
			
			//If a close parenthesis is found, a node ends so the current depth is decremented
			case ')':
				currDepth--;
				pos++;
				break;
				
			//If any other character, just go to next position in the String
			default:
				pos++;
				break;
			}
		}while(ind.charAt(pos) != ';'); //Finish when the end character ";" is found
		
		return ind;
	}
	
	/**
	 * Creates a child in the form (C ... C) including n times the 'C'
	 * 
	 * For calculating normal distribution: https://www.uv.es/ceaces/scrips/tablas/tanormal.htm
	 * 
	 * @param n Number of children
	 * @return String
	 */
	private String child(int n) {
		//The threshold is calculated with a gaussian, where the probability to be between 0 and 1 is 99.9%
		double threshold = utils.randomThreshold(gaussianStdv);
		
		//Start with open parenthesis, 'v' (for voting), and threshold ('.\d+')
		String child = "(v." + String.format(Locale.US, "%.2f", threshold).split("\\.")[1] + " ";
		
		//Add n times the "C" and a space
		for(int i=0; i<n; i++) {
			child += "C ";
		}
		
		//Replace last space " " with end parenthesis
		child = child.substring(0, child.length()-1) + ")";
		
		return child;
	}
	
	/**
	 * Creates a child in the form (C ... C) including the 'C' a random number of times between [2, max]
	 * 
	 * @param minChildren Min number of children
	 * @param maxChildren Max number of children
	 * @return String
	 */
	private String childRandomSize(int minChildren, int maxChildren) {
		return child(randgen.choose(minChildren, maxChildren+1)); //between min and max, included
	}
	
	/**
	 * Replace a given position in the string with a new string
	 * 
	 * @param str Original string 
	 * @param pos Position to replace
	 * @param newStr String to insert in position 'pos'
	 * @return Modified string
	 */
	public String replace(String str, int pos, String newStr) {
		return str.substring(0, pos) + newStr + str.substring(pos+1, str.length());
	}
}
