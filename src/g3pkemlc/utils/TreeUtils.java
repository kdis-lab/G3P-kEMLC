package g3pkemlc.utils;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import g3pkemlc.Prediction;

/**
 * Class implementing some utils of the G3P-kEMLC Tree individuals, such as obtain final prediction, or number of votes of each label.
 * 
 * @author Jose M. Moyano
 *
 */
public class TreeUtils {

	/**
	 * Reduce the tree and obtain the final tree prediction
	 * 
	 * @param ind Individual, tree
	 * @param key Key for the table according to corresponding individual being evaluated (used for concurrency)
	 * @param tablePredictions Table with predictions of other classifiers and combinations of them
	 * @param nInstances Number of instances 
	 * @param useConfidences True if confindences are used to combine predictions; otherwise bipartitions are used
	 * @return Combined prediction of all nodes in the tree
	 */
	public static Prediction reduce(String ind, String key, Hashtable<String, Prediction> tablePredictions, int nInstances, boolean useConfidences) {
		//Match two or more leaves (numer or _number) between parenthesis
		Pattern pattern = Pattern.compile("\\((_?\\d+ )+_?\\d+\\)");
		Matcher m = pattern.matcher(ind);
		
		//count to add predictions of combined nodes into the table
		int count = 0;
		Prediction pred = null;// = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
		while(m.find()) {
			//Combine the predictions of current nodes
			pred = combine(m.group(0), key,tablePredictions, nInstances, useConfidences);
			
			//Put combined predictions into the table
			tablePredictions.put(key+"_"+count, pred);
			
			//Replace node in the genotype
			ind = ind.substring(0, m.start()) + "_" + count + ind.substring(m.end(), ind.length());
			
			//Increment counter and match next one
			count++;
			m = pattern.matcher(ind);
		}
		
		//Return final prediction
		//	It is the combination of all nodes in level 1
		return pred;
	}
	
	
	/**
	 * Combine the predictions of several nodes
	 * 
	 * @param nodes String with the nodes to combine
	 * @param key Key for the table according to corresponding individual being evaluated (used for concurrency)
	 * @param tablePredictions Table with predictions of other classifiers and combinations of them
	 * @param nInstances Number of instances 
	 * @param useConfidences True if confindences are used to combine predictions; otherwise bipartitions are used
	 * @return Combined prediction of given nodes
	 */
	protected static Prediction combine(String nodes, String key, Hashtable<String, Prediction> tablePredictions, int nInstances, boolean useConfidences){
		Prediction pred = new Prediction(nInstances);
		
		Pattern pattern = Pattern.compile("\\d+");
		Matcher m;
		
		int n;
		
		//Split the nodes by space, so get the leaves
		String [] pieces = nodes.split(" ");
		for(String piece : pieces) {
			//Get index included in the piece and store in n
			m = pattern.matcher(piece);
			m.find();
			n = Integer.parseInt(m.group(0));

			//If the piece contains the character "_" is a combination of other nodes
			if(piece.contains("_")) {
				//Add to the current prediction, the prediction of one of the previously combined nodes
				pred.addPrediction(tablePredictions.get(key+"_"+n));
				
				//Remove because it is not going to be used again
				tablePredictions.replace(key+"_"+n, new Prediction());
			}
			else {
				//Add to the current prediction, the prediction of the corresponding classifier
				pred.addPrediction(tablePredictions.get(String.valueOf(n)));
			}
		}
		
		//Divide prediction by the number of learners and apply threshold if applicable
		if(useConfidences) {
			pred.divide();
		}
		else {
			pred.divideAndThresholdPrediction((float)0.5);
		}

		//Return combined prediction of corresponding nodes
		return pred;
	}
	
	/**
	 * Calculate the votes per label given a tree individual
	 * 
	 * @param ind Individual (tree)
	 * @param klabelsets Array with all k-labelsets
	 * @param nLabels Total number of labels
	 * @return Array with the number of votes per label
	 */
	public static int[] votesPerLabel(String ind, ArrayList<KLabelset> klabelsets, int nLabels) {
		int[] votes = new int[nLabels];
		
		Utils utils = new Utils();
		ArrayList<Integer> leaves = utils.getLeaves(ind);
		
		for(Integer leaf : leaves) {
			for(Integer label : klabelsets.get(leaf).getKlabelset()) {
				votes[label]++;
			}
		}
		
		return votes;
	}
	
	/**
	 * Calculate the average number of votes per label
	 * 
	 * @param votesPerLabel Array with the number of votes per label
	 * @return Average number of votes per label
	 */
	public static double avgVotes(int[] votesPerLabel) {
		double sum = 0.0;
		
		for(int i=0; i<votesPerLabel.length; i++) {
			sum += votesPerLabel[i];
		}
		
		return sum/votesPerLabel.length;
	}
}
