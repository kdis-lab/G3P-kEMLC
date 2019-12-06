package gpemlc;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import gpemlc.utils.DatasetTransformation;
import gpemlc.utils.KLabelset;
import gpemlc.utils.Utils;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.TechnicalInformation;

/**
 * Class implementing the ensemble for the GP-EMLC algorithm.
 * 
 * @author Jose M. Moyano
 *
 */
public class EMLC extends MultiLabelMetaLearner {

	/**
	 * serialVersionUID
	 */
	private static final long serialVersionUID = 3236560089372633198L;
	
	/**
	 * Genotype of the tree
	 */
	String genotype;
	
	/**
	 * List of integer indicating the leaves in the individual
	 */
	ArrayList<Integer> leaves;
	
	/**
	 * Classifiers of the ensemble
	 */
	Hashtable<String, MultiLabelLearnerBase> learners;
	
	/**
	 * Array of k-labelsets
	 */
	ArrayList<KLabelset> klabelsets;
	
	/**
	 * Table with predictions of combined nodes
	 */
	Hashtable<String, Prediction> tablePredictions = new Hashtable<String, Prediction>();
	
	/**
	 * Utils
	 */
	Utils utils = new Utils();
	
	/**
	 * Determine if uses confidences or bipartitions to combine predictions
	 */
	boolean useConfidences;

	/**
	 * Label indices of all labels in the data
	 */
	int[] labelIndices;
	
	/**
	 * Constructor
	 * 
	 * @param baseLearner Multi-label base learner
	 */
	public EMLC(MultiLabelLearner baseLearner) {
		super(baseLearner);
	}
	
	/**
	 * Constructor with genotype
	 * 
	 * @param baseLearner Multi-label base learner
	 * @param genotype Genotype of the individual tree
	 */
	public EMLC(MultiLabelLearner baseLearner, String genotype) {
		super(baseLearner);
		this.genotype = genotype;
		this.klabelsets = null;
		this.leaves = utils.getLeaves(genotype);
		this.useConfidences = false;
	}	
	
	/**
	 * Constructor with genotype and useConfidences
	 * 
	 * @param baseLearner Multi-label base learner
	 * @param genotype Genotype of the individual tree
	 * @param useConfidences true if confindences are used to combine predictions, false otherwise
	 */
	public EMLC(MultiLabelLearner baseLearner, ArrayList<KLabelset> klabelsets, String genotype, boolean useConfidences) {
		super(baseLearner);
		this.klabelsets = klabelsets;
		this.genotype = genotype;
		this.leaves = utils.getLeaves(genotype);
		this.useConfidences = useConfidences;
	}	

	/**
	 * Reset seed for each member
	 */
	public void resetSeed(int seed) {
		for(String key : learners.keySet()) {
			((LabelPowerset2)learners.get(key)).setSeed(seed);
		}
	}
	
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		learners = new Hashtable<String, MultiLabelLearnerBase>(leaves.size());
		
		//Load each learner from hard disk
		for(int i=0; i<leaves.size(); i++) {
			learners.put(String.valueOf(leaves.get(i)), (MultiLabelLearnerBase) utils.loadObject("mlc/classifier"+leaves.get(i)+".mlc"));
		}
		
		//Store the label indices of the original dataset
		labelIndices = trainingSet.getLabelIndices();
	}
	
	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
		//Get final prediction by reducing the tree
		double [] confs = reduce(genotype, instance).pred[0];
		
		//Transform to multi-label output
		boolean[] bip = new boolean[numLabels];
		for(int i=0; i<numLabels; i++) {
			if(confs[i] >= 0.5) {
				bip[i] = true;
			}
			else {
				bip[i] = false;
			}
		}
		
		return new MultiLabelOutput(bip, confs);
	}
	
	/**
	 * Reduce the tree and obtain the final tree prediction for a given instance
	 * 
	 * @param ind Individual, tree
	 * @param instance Instance to predict
	 * @return Prediction of the tree for the instance
	 */
	public Prediction reduce(String ind, Instance instance) {
		//Match two or more leaves (numer or _number) between parenthesis
		Pattern pattern = Pattern.compile("\\((_?\\d+ )+_?\\d+\\)");
		Matcher m = pattern.matcher(ind);
		
		//count to add predictions of combined nodes into the table
		int count = 0;
		Prediction pred = new Prediction(1); //(1, numLabels)
		
		while(m.find()) {
			//Combine the predictions of current nodes
			pred = combine(m.group(0), instance);
			
			//Put combined predictions into the table
			tablePredictions.put("_"+count, pred);
			
			//Replace node in the genotype
			ind = ind.substring(0, m.start()) + "_" + count + ind.substring(m.end(), ind.length());
			
			//Increment counter and match next one
			count++;
			m = pattern.matcher(ind);
		}
		
		return pred;
	}
	
	/**
	 * Combine predictions of several nodes
	 * 
	 * @param nodes String with the nodes to combine
	 * @param instance Instance to predict
	 * @return Combined prediction
	 */
	protected Prediction combine(String nodes, Instance instance){
		Prediction pred = new Prediction(1); //null; //new Prediction(1); //(1, numLabels)
		
		Pattern pattern = Pattern.compile("\\d+");
		Matcher m;
		
		int n;
		
		DatasetTransformation dt = new DatasetTransformation();
		
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
				pred.addPrediction(tablePredictions.get("_"+n));
				
				//Remove because it is not going to be used again
				tablePredictions.remove("_"+n);
			}
			else {
				//Add to the current prediction, the prediction of the corresponding classifier
				pred.addPrediction(dt.getOriginalLabelIndices(labelIndices, klabelsets.get(n).getKlabelset()), getPredictions(learners.get(String.valueOf(n)), instance));
			}
		}

		//Divide prediction by the number of learners and apply threshold
		if(useConfidences) {
			pred.divide();
		}
		else {
			pred.divideAndThresholdPrediction(0.5);
		}
		
		return pred;
	}

	/**
	 * Get the predictions of a given learner and for a given instance
	 * 
	 * @param learner Multi-label classifier
	 * @param instance Instance to predict
	 * @return Predictions
	 */
	protected double[][] getPredictions(MultiLabelLearner learner, Instance instance){
		double[][] pred = new double[1][numLabels];
		pred[0] = null;
		
		try {
			if(useConfidences) {
				pred[0] = learner.makePrediction(instance).getConfidences();
			}
			else {
				pred[0] = utils.bipartitionToConfidence(learner.makePrediction(instance).getBipartition());
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return pred;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}

}
