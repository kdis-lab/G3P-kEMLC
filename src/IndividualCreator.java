import java.util.Random;

/**
 * Class implementing the individual creator for GP-EMLC individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class IndividualCreator {
	
	/**
	 * Random numbers generator
	 */
	static Random rand = new Random();
	
	/**
	 * Create an individual of given maximum depth
	 * 
	 * @param nMax Max value for the leaves
	 * @param maxDepth Maximum depth of the tree
	 * @return Individual as String
	 */
	public String create(int nMax, int maxDepth) {
		String ind = "S;";
		int pos = 0;
		int currDepth = 0;
		
		do {
			switch (ind.charAt(pos)) {
			case 'S':
				ind = replace(ind, pos, "(C C C)");
				break;
			case 'C':
				if(currDepth < maxDepth) {
					if(rand.nextBoolean()) {
						ind = replace(ind, pos, "(C C C)");
					}
					else {
						ind = replace(ind, pos, String.valueOf(rand.nextInt(nMax)));
					}
				}
				else {
					ind = replace(ind, pos, String.valueOf(rand.nextInt(nMax)));
				}				
				break;
			
			case '(':
				currDepth++;
				pos++;
				break;
			case ')':
				currDepth--;
				pos++;
				break;
				
			default:
				pos++;
				break;
			}
		}while(ind.charAt(pos) != ';');
		
		return ind;
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
