package g3pkemlc;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;

/**
 * Class implementing the predictions of a given classifier
 * 
 * @author Jose M. Moyano
 *
 */
public class Prediction {
	
	/**
	 * Number of instances in the prediction
	 */
	public int nInstances;
	
	/**
	 * Indices of labels included in the prediction
	 */
	ArrayList<Integer> labelIndices;
	
	/**
	 * Number of votes for each of the labels
	 */
	public int[] labelVotes;
	
	/**
	 * Prediction (double to be able to store confidences)
	 */
	public double[][] pred;
	
	/**
	 * Default constructor
	 */
	public Prediction() {
		nInstances = -1;
		labelIndices = null;
		labelVotes = null;
		pred = null;
	}
	
	/**
	 * Constructor with parameters
	 * 
	 * @param nInstances Number of instances
	 */
	public Prediction(int nInstances) {
		this.nInstances = nInstances;
		this.labelIndices = new ArrayList<Integer>(0);
		this.labelVotes = new int[0];
		this.pred = new double[nInstances][];
	}
	
	/**
	 * Constructor with parameters
	 * 
	 * @param labelIndices Indices (of the original dataset) of the labels included in the prediction
	 * @param prediction Prediction of the classifier (allows confidence values)
	 */
	public Prediction(int[] labelIndices, double[][] prediction) {
		this.nInstances = prediction.length;
		
		this.labelIndices = new ArrayList<Integer>(labelIndices.length);
		for(int i=0; i<labelIndices.length; i++) {
			this.labelIndices.add(labelIndices[i]);
		}
		
		this.labelVotes = new int[labelIndices.length];
		for(int i=0; i<labelIndices.length; i++) {
			labelVotes[i] = 1;
		}
		
		this.pred = prediction.clone();
	}
	
	/**
	 * Copy constructor
	 * 
	 * @param prediction Prediction object to copy
	 */
	public Prediction(Prediction prediction) {
		this.nInstances = prediction.nInstances;
		this.labelIndices = new ArrayList<Integer>();
		labelIndices.addAll(prediction.labelIndices);
		this.labelVotes = prediction.labelVotes.clone();
		this.pred = prediction.pred.clone();
	}
	
	/**
	 * Transform the predictions as confidences into bipartitions given a threshold
	 * 
	 * @param threshold Threshold to determine relevant and irrelevant labels
	 * @return Bipartitions boolean matrix with bipartitions
	 */
	public boolean[][] getBipartition(double threshold) {
		boolean[][] bipartition = new boolean[nInstances][labelIndices.size()];
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<labelIndices.size(); j++) {
				if(this.pred[i][j] >= threshold) {
					bipartition[i][j] = true;
				}
				else {
					bipartition[i][j] = false;
				}
			}
		}
		
		return bipartition;
	}
	
	/**
	 * Transform the predictions as confidences for a given instance into bipartitions given a threshold
	 * 
	 * @param instance Instance to transform prediction into bipartition
	 * @param threshold Threshold to determine relevant and irrelevant labels
	 * @return Bipartitions
	 */
	public boolean[] getBipartition(int instance, double threshold) {
		boolean[] bipartition = new boolean[labelIndices.size()];
		
		for(int j=0; j<labelIndices.size(); j++) {
			if(this.pred[instance][j] >= threshold) {
				bipartition[j] = true;
			}
			else {
				bipartition[j] = false;
			}
		}
		
		return bipartition;
	}
	
	/**
	 * Add a prediction to the current one
	 * 
	 * @param other Prediction to add to the current one
	 */
	public void addPrediction(Prediction other) {
		//Check if both predictions are made for the same number of instances
		if(this.nInstances != other.nInstances) {
			System.out.println("The number of instances is not the same in both predictions.");
			System.exit(-1);
		}
		
		//Combine the label indices in an new array
		ArrayList<Integer> newLabelIndices = new ArrayList<Integer>(this.labelIndices);
		
		for(int i=0; i<other.labelIndices.size(); i++) {
			if(! newLabelIndices.contains(other.labelIndices.get(i))) {
				newLabelIndices.add(other.labelIndices.get(i));
			}
		}
		
		//Sort label indices
		Collections.sort(newLabelIndices);
		
		//Create arrays for new labelVotes and new predictions
		int[] newLabelVotes = new int[newLabelIndices.size()];
		double[][] newPred = new double[this.nInstances][newLabelIndices.size()];
		
		int currLabelIndex;
		//For each label index in any of the predictions (i.e., in newLabelIndices)
		for(int l=0; l<newLabelIndices.size(); l++) {
			//Get the current label index to work with it
			currLabelIndex = newLabelIndices.get(l);
			
			//If both contain the same label, combine predictions
			if(this.labelIndices.contains(currLabelIndex) && other.labelIndices.contains(currLabelIndex)) {
				int thisLabelPos = this.labelIndices.indexOf(currLabelIndex);
				int otherLabelPos = other.labelIndices.indexOf(currLabelIndex);
				
				for(int i=0; i<nInstances; i++) {
					newPred[i][l] = this.pred[i][thisLabelPos] + other.pred[i][otherLabelPos];
					newLabelVotes[l] = this.labelVotes[thisLabelPos] + other.labelVotes[otherLabelPos];
				}
			}
			//If only *this* contains the label; just copy this predictions
			else if(this.labelIndices.contains(currLabelIndex)) {
				int thisLabelPos = this.labelIndices.indexOf(currLabelIndex);
				for(int i=0; i<nInstances; i++) {
					newPred[i][l] = this.pred[i][thisLabelPos];
					newLabelVotes[l] = this.labelVotes[thisLabelPos];
				}
			}
			//If only *other* contains the label; just copy other predictions
			else if(other.labelIndices.contains(currLabelIndex)) {
				int otherLabelPos = other.labelIndices.indexOf(currLabelIndex);
				for(int i=0; i<nInstances; i++) {
					newPred[i][l] = other.pred[i][otherLabelPos];
					newLabelVotes[l] = other.labelVotes[otherLabelPos];
				}
			}
			else {
				System.out.println("An error ocurred when combining predictions");
				System.exit(-1);
			}
		}
		
		//Copy the new labelIndices, labelVotes, and predictions to the current object
		this.labelIndices = new ArrayList<Integer>(newLabelIndices);
		this.labelVotes = newLabelVotes.clone();
		this.pred = newPred.clone();
	}
	
	/**
	 * Add a prediction to the current one
	 * This method is deprecated, going to be deleted.
	 * 	I still need to see where I'm using it and solve it
	 * 
	 * @param other Prediction to add to the current one
	 */
	public void addPrediction(int[] labelIndices, double[][] pred) {
		this.addPrediction(new Prediction(labelIndices, pred));
	}
		
	/**
	 * Divide the current prediction by the number of votes of each label and apply the threshold.
	 * If each prediction is lower than the thresohld, it is negative; and positive otherwise.
	 * 
	 * @param threshold Threshold
	 */
	public void divideAndThresholdPrediction(double threshold) {
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<labelIndices.size(); j++) {
				if((this.pred[i][j])/labelVotes[j] >= threshold) {
					this.pred[i][j] = 1;
				}
				else {
					this.pred[i][j] = 0;
				}
			}
		}
		
		for(int j=0; j<labelVotes.length; j++) {
			labelVotes[j] = 1;
		}
	}
	
	/**
	 * Divide the current prediction by the number of votes of each label
	 */
	public void divide() {
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<labelIndices.size(); j++) {
				this.pred[i][j] /= labelVotes[j];
			}
		}
		
		for(int j=0; j<labelVotes.length; j++) {
			labelVotes[j] = 1;
		}
	}
	
	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat("#.###");
		
		String s = "";
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<labelIndices.size(); j++) {
				s += df.format(this.pred[i][j]) + ", ";
			}
			s += "\n";
		}
	
		return s;
	}
	
}
