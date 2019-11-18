package gpemlc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing several utilities to deal with GP-EMLC individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class Utils {
	
	/**
	 * Random numbers generator
	 */
	public static IRandGen randgen = null;

	/**
	 * Empty constructor
	 */
	public Utils() {
		//Do nothing.
		//At least once the Utils class need to be instantiated with randgen
		//Then we can just use this constructor
	}
	
	/**
	 * Constructor with random numbers generator
	 * 
	 * @param rand Random numbers generator
	 */
	public Utils(IRandGen randgen) {
		Utils.randgen = randgen;
	}
	
	/**
	 * Counts the parenthesis (therefore possible subtrees) of the tree
	 * 
	 * @param ind Individual
	 * @return Number of begin parenthesis
	 */
	public int countParenthesis(String ind) {
		return ((int) ind.chars().filter(ch -> ch == '(').count());
	}
	
	/**
	 * Counts the number of leaves in the individual
	 * 
	 * @param ind Individual
	 * @return Number of leaves
	 */
	public int countLeaves(String ind) {
		Pattern p = Pattern.compile("\\d+"); // "\d" is for digits in regex
		Matcher m = p.matcher(ind);
		int count = 0;
		
		//Increment the counter each time that a number is found on the String
		while(m.find()){
			count++;
		}
		
		return count;
	}
	
	/**
	 * Get a list including the leaves of the tree (without repetition)
	 * 
	 * @param ind Tree
	 * @return ArrayList of Integers with leaves without repetition
	 */
	public ArrayList<Integer> getLeaves(String ind) {
		Pattern p = Pattern.compile("\\d+"); // "\d" is for digits in regex
		Matcher m = p.matcher(ind);
		
		Integer leaf;
		ArrayList<Integer> leaves = new ArrayList<Integer>();
		
		//Add to the list each different number (leaf) found
		while(m.find()){
			leaf = Integer.parseInt(m.group());
			if(!leaves.contains(leaf)) {
				leaves.add(leaf);
			}
		}
		
		Collections.sort(leaves);
		
		return leaves;
	}
	
	/**
	 * Choose a random leaf
	 * 
	 * @param ind Individual
	 * @return Array with index of start and end of the current leaf
	 */
	public int[] chooseRandomLeaf(String ind) {
		//Get the number of leaves
		int nLeaves = countLeaves(ind);
		int r = randgen.choose(0, nLeaves);
		
		Pattern p = Pattern.compile("\\d+"); // "\d" is for digits in regex
		Matcher m = p.matcher(ind);
		
		//Match leaves until counter reaches the randomly generated r
		int count = 0;
		do {
			m.find();
			count++;
		}while(count<=r);
		
		//Indexes of start and end of the leaf in the String
		int[] leaf = new int[2];
		leaf[0] = m.start();
		leaf[1] = m.end();

		return leaf;
	}
	
	/**
	 * Calculate the depth of a given node in the tree
	 * 
	 * @param ind Individual
	 * @param nodeStart Index of the string where the node starts
	 * @return Depth of the node
	 */
	public int calculateNodeDepth(String ind, int nodeStart) {
		int depth = 0;
		
		for(int i=0; i<nodeStart; i++) {
			switch (ind.charAt(i)) {
			case '(':
				depth++;
				break;

			case ')':
				depth--;
				break;
			}
		}
		
		return depth;
	}
	
	/**
	 * Calculate the maximum depth of any node in the tree
	 * 
	 * @param tree Tree
	 * @return Maximum depth
	 */
	public int calculateTreeMaxDepth(String tree) {
		int currDepth = 0, maxDepth = -1;
		
		for(int i=0; i<tree.length(); i++) {
			switch (tree.charAt(i)) {
			case '(':
				currDepth++;
				if(currDepth > maxDepth) {
					maxDepth = currDepth;
				}
				break;

			case ')':
				currDepth--;
				break;
			}
		}
		
		return maxDepth;
	}
	
	/**
	 * Choose a random subtree of a given individual
	 * 
	 * @param ind Individual
	 * @param considerRoot True if the whole tree is considered as a feasible subtree and false otherwise
	 * @return Array with start and end indexes of the subtree
	 */
	public int[] chooseRandomSubTree(String ind, boolean considerRoot) {
		int parenthesis = countParenthesis(ind);
		int [] subTree = new int[2];
		int r;
		
		if(considerRoot || parenthesis > 1) {
			//If the whole tree is not considered as a subtree, avoid to select the first parenthresis
			if(!considerRoot) {
				r = randgen.choose(1, parenthesis); //avoid to select root (parenthesis 0)
			}
			else {
				r = randgen.choose(0, parenthesis);
			}
			
			int beginParenthesisCount = 0, extraBeginParenthesis = 0;
			int startSubstr=0, endSubstr=0;
			boolean extracting = false, found = false;
			int pos = 0;
			do {
				switch (ind.charAt(pos)) {
				case '(':
					//If we are not yet extracting a subtree, check if we want to extract current subtree
					if(!extracting) {
						if(beginParenthesisCount == r) {
							extracting = true;
							startSubstr = pos;
						}
						else {
							beginParenthesisCount++;
						}
					}
					//If we are already extracting subtree, consider that inside there are a new parenthesis
						//that need to be closed first
					else {
						extraBeginParenthesis++;
					}
					
					break;
				case ')':
					//If we are extracting a subtree, check if we have closed all sub-node parenthesis
					if(extracting) {
						if(extraBeginParenthesis > 0) {
							extraBeginParenthesis--;
						}
						else {
							endSubstr = pos;
							found = true;
						}
					}
					break;
					
				default:
					break;
				}
				pos++;
			}while(!found);
			
			subTree[0] = startSubstr;
			subTree[1] = endSubstr+1;
		}
		else {
			return null;
		}
		
		return subTree;
	}
	
	/**
	 * Check if an individual is correct or not
	 * 
	 * @param ind Individual
	 * @return True if it is feasible and correct; false otherwise
	 */
	public boolean checkInd(String ind) {
		Pattern pattern = Pattern.compile("\\((\\d+ )+\\d+\\)");
		
		//Replace each match by "0"
		while(pattern.matcher(ind).find()) {
			ind = ind.replaceAll("\\((\\d+ )+\\d+\\)", "0");
		}
		
		//If the reduced individual matchs with "0;", it is feasible
		if(ind.equals("0;")) {
			return true;
		}
		else {
			return false;
		}
	}
	
	/**
	 * Check if a given node is a leaf (if it just contains a number)
	 * 
	 * @param node Node of the tree
	 * @return True if it is a leaf; false otherwise
	 */
	public boolean isLeaf(String node) {
		return node.matches("^\\d+$");
	}
	
	/**
	 * Check if a given String correspond to a full node
	 * 	A node start and end with parenthesis, and may end with ";"
	 * 
	 * @param node String containing a possible node
	 * @return True if it is a node; false otherwise
	 */
	public boolean isNode(String node) {
		return node.matches("^\\(.*\\)(;)?$");
	}
	
	/**
	 * Write an object to the hard disk
	 * 
	 * @param obj Object to write
	 * @param filepath Path of the file for the object
	 */
	public void writeObject(Object obj, String filepath) {
        try {
            FileOutputStream fileOut = new FileOutputStream(filepath);
            ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
            objectOut.writeObject(obj);
            objectOut.close();
 
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
	
	/**
	 * Load an object from hard disk
	 * 
	 * @param filepath Path of the file
	 * @return Loaded object
	 */
	public Object loadObject(String filepath) {
        Object obj = null;
		
		try {
        	FileInputStream fi = new FileInputStream(new File(filepath));
			ObjectInputStream oi = new ObjectInputStream(fi);

			obj = oi.readObject();
			oi.close();
 
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        
        return obj;
    }
	
}