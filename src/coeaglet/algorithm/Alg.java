package coeaglet.algorithm;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.configuration.Configuration;

import coeaglet.individualCreator.FrequencyBasedIndividualCreator;
import coeaglet.mutator.Mutator;
import coeaglet.mutator.SubpopMutator;
import coeaglet.recombinator.Crossover;
import coeaglet.utils.Utils;
import coeaglet.utils.Utils.CommunicationType;
import coeaglet.utils.Utils.EvalType;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.core.MulanException;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.classic.MultiSGE;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.listind.MultipListCreator;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.selector.WorsesSelector;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;

/**
 * Class implementing the main co-evolutionary algorithm for the optimization of multi-label ensembles.
 * It is based on a Multiple populations algorithm (MultiSGE)
 * 
 * CoEAGLET. More information at: https://github.com/i02momuj/CoEAGLET
 * 
 * @author Jose M. Moyano
 *
 */
public class Alg extends MultiSGE {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 8988750329823184523L;
	
	/**
	 * Number of labels of the dataset
	 */
	int nLabels;
	
	/**
	 *  Number of active labels at each base classifier
	 */
	private int k;
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Training datasets
	 */
	MultiLabelInstances [] trainData;
	
	/**
	 * Test dataset
	 */
	MultiLabelInstances testData;
	
	/**
	 * Table including the fitness of all individuals
	 */
	Hashtable<String, Double> tableFitness = new Hashtable<String, Double>();
	
	/**
	 * Table including all built classifiers
	 */
	Hashtable<String, MultiLabelLearner> tableClassifiers = new Hashtable<String, MultiLabelLearner>();
	
	/** 
	 * Betters selector.
	 */	
	public BettersSelector bettersSelector = new BettersSelector(this);
	
	/**
	 * Worses selector.
	 */
	private WorsesSelector worseSelector = new WorsesSelector(this);
	
	/**
	 * Generator of random numbers
	 */
	IRandGen randgen;
	
	/**
	 * Best ensemble generated so far
	 */
	Ensemble bestEnsemble;
	
	/**
	 * Fitness of bestEnsemble
	 */
	double bestEnsembleFitness = -1;
	
	/**
	 * Current ensemble fitness (for listener purposes)
	 */
	double currentEnsembleFitness;
	
	/**
	 * Ratio of instances sampled at each train data
	 */
	double sampleRatio;
	
	/**
	 * Evalaution typa of individuals in the population
	 */
	EvalType evalType;
	
	/**
	 * Type of communication used in the algorithm
	 */
	CommunicationType commType;
	
	/**
	 * Crossover probability in operators communication 
	 */
	double probCrossComm;
	
	/**
	 * Mutation probability in operators communication
	 */
	double probMutComm;
	
	/**
	 * Threshold for ensemble prediction
	 */
	double predictionThreshold;
	
	/**
	 * Number of classifiers in the ensemble
	 */
	int nClassifiers;
	
	/**
	 * Beta value to update subpopulations
	 */
	double betaUpdatePop;
	
	/**
	 * Beta value to build the ensemble
	 */
	double betaEnsembleSelection;
	
	/**
	 * Indicates if the ensemble is pruned or not in the last iteration
	 */
	boolean prune;
	
	/**
	 * Parameter for the exponential function for the exchange communication. By default omega = 10
	 * The lower the value, the higher the probability of communication
	 */
	double omega = 10;

	
	/**
	 * Constructor
	 */
	public Alg()
	{
		super();
	}
	
	/**
	 * Gets the max number of labels per classifier
	 * 
	 * @return Max number of labels per classifier
	 */
	public int getk()
	{
		return k;
	}
	
	/**
	 * Getter for the ensemble
	 * 
	 * @return Ensemble
	 */
	public Ensemble getEnsemble() {
		return bestEnsemble;
	}
	
	/**
	 * Getter for array of training data
	 * 
	 * @return Array with all sampled training datasets
	 */
	public MultiLabelInstances[] getTrainData() {
		return trainData;
	}
	
	/**
	 * Getter for an specific training data
	 *  
	 * @param p Identifier of subpopulation for the training data 
	 * @return Training data for the corresponding subpopulation
	 */
	public MultiLabelInstances getTrainData(int p) {
		return trainData[p];
	}
	
	/**
	 * Getter for full training dataset
	 * 
	 * @return Full training dataset
	 */
	public MultiLabelInstances getFullTrainData() {
		return fullTrainData;
	}
	
	/**
	 * Getter for test data
	 * 
	 * @return Test dataset
	 */
	public MultiLabelInstances getTestData() {
		return testData;
	}
	
	/**
	 * Getter for current ensemble fitness
	 * 
	 * @return Fitness of the current ensemble
	 */
	public double getCurrentEnsembleFitness() {
		return currentEnsembleFitness;
	}
	
	/**
	 * Get best individual of each subpopulation
	 * 
	 * @return List with best individual of each subpopulation
	 */
	public List<MultipListIndividual> bestIndividuals(){
		List<MultipListIndividual> list = new ArrayList<MultipListIndividual>(numSubpop);
		
		for(int i=0; i<numSubpop; i++) {
			list.add((MultipListIndividual) bettersSelector.select(bset.get(i), 1).get(0));
		}
		
		return list;
	}
	
	/**
	 * Get the average fitness of the individuals of each subpopulation
	 * 
	 * @return Array with average fitness of each subpopulation
	 */
	public double[] avgFitnessSubpopulation() {
		double [] avgFit = new double[numSubpop];
		
		for(int p=0; p<numSubpop; p++) {
			avgFit[p] = 0;
			for(int i=0; i<subpopSize; i++) {
				avgFit[p] += ((SimpleValueFitness)bset.get(p).get(i).getFitness()).getValue();
			}
		}
		
		for(int p=0; p<numSubpop; p++) {
			avgFit[p] /= subpopSize;
		}
		
		return avgFit;
	}
	
	
	
	/**
	 * Configure some default aspects and parameters of EME to make the configuration easier
	 * 
	 * @param configuration Configuration
	 */
	private void configureEagletDefaults(Configuration configuration) {
		//Species
		configuration.setProperty("species[@type]", "net.sf.jclec.listind.MultipListIndividualSpecies");
		configuration.setProperty("species[@genotype-length]", "1");
		
		//Provider (only if not provided)
		if(! configuration.containsKey("provider[@type]")) {
			configuration.addProperty("provider[@type]", "net.sf.jclec.listind.MultipListCreator");
		}
		
		//Randgen type (only if not provided)
		if(! configuration.containsKey("rand-gen-factory[@type]")) {
			configuration.addProperty("rand-gen-factory[@type]", "net.sf.jclec.util.random.RanecuFactory");
		}
		
		//Parents-selector (only if not provided)
		if(! configuration.containsKey("parents-selector[@type]")) {
			configuration.addProperty("parents-selector[@type]", "net.sf.jclec.selector.TournamentSelector");
		}
		if(! configuration.containsKey("parents-selector.tournament-size")) {
			configuration.addProperty("parents-selector.tournament-size", "2");
		}
		
		//Listener type (only if not provided)
		if(! configuration.containsKey("listener[@type]")) {
			configuration.addProperty("listener[@type]", "net.sf.jclec.listener.PopulationReporter");
		}
		
		//Prediction treshold
		if(! configuration.containsKey("prediction-threshold")) {
			configuration.addProperty("prediction-threshold", "0.5");
		}
	}
	
	
	@Override
	public void configure(Configuration configuration)
	{
		configureEagletDefaults(configuration);
		super.configure(configuration);
		
		randgen = randGenFactory.createRandGen();

		k = configuration.getInt("k");
		predictionThreshold = configuration.getDouble("prediction-threshold");
		nClassifiers = configuration.getInt("number-classifiers");
		betaUpdatePop = configuration.getDouble("beta-update-population");
		betaEnsembleSelection = configuration.getDouble("beta-ensemble-selection");
		prune = configuration.getBoolean("prune-ensemble");

		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
		String commTypeString = configuration.getString("communication");
		switch (commTypeString) {
		case "no":
			commType = CommunicationType.no;					
			break;
		case "exchangeEnsemble":
			commType = CommunicationType.exchangeEnsemble;
			if(! configuration.containsKey("omega")) {
				omega = 2;
			}
			else {
				omega = configuration.getDouble("omega");
			}
			break;
		case "exchangeSubpop":
			commType = CommunicationType.exchangeSubpop;
			if(! configuration.containsKey("omega")) {
				omega = 2;
			}
			else {
				omega = configuration.getDouble("omega");
			}
			break;
		case "operators":
			commType = CommunicationType.operators;
			probCrossComm = configuration.getDouble("probability-crossover-communication");
			probMutComm = configuration.getDouble("probability-mutator-communication");
			break;
		}
		
		String evalTypeString = configuration.getString("eval-type");
		switch (evalTypeString) {
		case "train":
			evalType = EvalType.train;
			break;
		case "full":
			evalType = EvalType.full;	
			break;
		}
		
		fullTrainData = null;
		testData = null;
		try {
			fullTrainData = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			testData = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);
			
			trainData = new MultiLabelInstances[numSubpop];
			for(int p=0; p<numSubpop; p++) {
				trainData[p] = Utils.sampleData(fullTrainData, sampleRatio, randgen);
			}
		}
		catch(MulanException e) {
			e.printStackTrace();
		}
		
		nLabels = fullTrainData.getNumLabels();
		
		((Mutator) mutator.getDecorated()).setMaxInt(nLabels);
		
		((MultipListCreator) provider).setMaxInt(nLabels);
		((MultipListCreator) provider).setK(k);
		
		((Eval) evaluator).setTrainData(trainData);
		if(evalType == EvalType.full) {
			((Eval) evaluator).setEvalData(fullTrainData);
		}
		else {
			((Eval) evaluator).setEvalData(null);
		}
		((Eval) evaluator).setEvalData(fullTrainData);
		((Eval) evaluator).setTableFitness(tableFitness);
		((Eval) evaluator).setTableClassifiers(tableClassifiers);
	}
	
	/**
	 * Create individuals in population, evaluating before start rest
	 * of evolution
	 */
	protected void doInit() 
	{		
		//Calculate individuals by subpopulation
		subpopSize = (int)Math.round((populationSize*1.0) / numSubpop);
		
		bset = new ArrayList<List<IIndividual>>(numSubpop);
		pset = new ArrayList<List<IIndividual>>(numSubpop);
		cset = new ArrayList<List<IIndividual>>(numSubpop);
		rset = new ArrayList<List<IIndividual>>(numSubpop);
		
		for(int p=0; p<numSubpop; p++) {
			//Initialize each population
			((FrequencyBasedIndividualCreator) provider).setSubpopId(p);
			((FrequencyBasedIndividualCreator) provider).setaMin(1);
			((FrequencyBasedIndividualCreator) provider).setAppearances(Utils.getAppearances(trainData[p]));
			bset.add(p, provider.provide(subpopSize));
			
			// Evaluate individuals
			//evaluator.evaluate(bset.get(p));
		}
		
		//Evaluate individuals of all subpopulations
		((MultipAbstractParallelEvaluator)evaluator).evaluateMultip(bset);

		// Do Control
		doControl();
	}
	
	
	@Override
	protected void doUpdate() {
		//Update each subpopulation
		for(int p=0; p<numSubpop; p++) {			
			//Add to rset all individuals from bset
			for(IIndividual bInd : bset.get(p)) {
				if(!Utils.contains(rset.get(p), (MultipListIndividual) bInd)) {
					rset.get(p).add(bInd);
				}
			}
			
			//Add to rset individuals from cset that are not already contained
			for(IIndividual bInd : cset.get(p)) {
				if(!Utils.contains(rset.get(p), (MultipListIndividual) bInd)) {
					rset.get(p).add(bInd);
				}
			}
			
			//Update subpopulation with ensemble selection procedure
			EnsembleSelection eSel = new EnsembleSelection(rset.get(p), subpopSize, nLabels, betaUpdatePop);
			eSel.setRandgen(randgen);
			eSel.selectEnsemble();
			bset.set(p, eSel.getEnsemble());
			eSel = null;
			
			//Clear rest of sets
			pset.get(p).clear();
			rset.get(p).clear();
			cset.get(p).clear();
		}
	}
	
	
	/**
	 * At the moment just generate and evaluate the ensemble
	 */
	protected void doCommunication() 
	{
		/*
		 * First, build ensemble
		 */
		//Join all individuals of all subpopulations
		List<IIndividual> allInds = new ArrayList<IIndividual>();
		for(int p=0; p<numSubpop; p++) {
			allInds.addAll(bset.get(p));
		}
			
		//Add individuals of the best ensemble with a probability
		//	The probability is higher in last generations and lower in earlier
		if(bestEnsemble != null) {
			for(IIndividual ind : bestEnsemble.inds) {
				if(randgen.coin((generation*1.0)/maxOfGenerations)) {
					if(!Utils.contains(allInds, (MultipListIndividual)ind)) {
						allInds.add(ind.copy());
					}
				}
			}
		}
			
		//Create ensemble considering all individuals
		EnsembleSelection eSel = new EnsembleSelection(allInds, nClassifiers, nLabels, betaEnsembleSelection);
		eSel.setRandgen(randgen);
		eSel.selectEnsemble();
		LabelPowerset2 learner = new LabelPowerset2(new J48());
		((LabelPowerset2)learner).setSeed(1);
		
		Ensemble currentEnsemble = new Ensemble(eSel.getEnsemble(), learner);
		currentEnsemble.setTableClassifiers(tableClassifiers);
		
		//Object no longer used
		eSel = null;
		
		try {
			currentEnsemble.build(fullTrainData);

			//Evaluate ensemble
			EnsembleEval eEval = new EnsembleEval(currentEnsemble, fullTrainData);
			currentEnsembleFitness = eEval.evaluate();
			eEval = null;
				
			System.out.println("Fitness iter " + generation + ": " + currentEnsembleFitness);
			//System.out.println("currentEnsemble  votes: " + Arrays.toString(eSel.labelVotes));
			
			if(currentEnsembleFitness > bestEnsembleFitness) {
				System.out.println("\tNew best fitness!");
				//System.out.println(currentEnsemble);
				bestEnsemble = currentEnsemble;
				bestEnsembleFitness = currentEnsembleFitness;
			}
				
			System.out.println();
				
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		/*
		 * If applicable, communicate subpops
		 */
		MultipListIndividual currInd, newInd;
		int sp, r;
		switch (commType) {
		case no:
			//Do nothing
			break;
			
		case exchangeEnsemble:
			/*
			 * For the better individuals in the ensemble, try to copy to other subpopulations
			 * The probability of copy is biased by their position in the ensemble
			 */
			double prob;
			int exchanged = 0;
			
			for(int i=0; i<currentEnsemble.inds.size(); i++) {
				//Probability to be selected is lower as it is deeper in the ensemble
				//It is calculated as an exponential (importance ^ omega)
					//The greater the omega value, the greater the slope, so deeper individuals have less probability
				prob = Math.pow(((1.0*nClassifiers - i) / nClassifiers), omega);

				if(randgen.coin(prob)) {
					currInd = currentEnsemble.inds.get(i);
					
					//Select subpop different to the current one
					sp = currInd.getSubpop();
					do {
						r = randgen.choose(0, numSubpop);
					}while(r == sp);
					
					//Individual with same list of genotype, but different subpopulation
					newInd = new MultipListIndividual(new MultipListGenotype(r, new ArrayList<Integer>(currInd.getGenotype().genotype)));
					
					//If it has been evaluated yet at any moment
						//Probability to be included again is reduced with generations
					//If not, just include
					//Only include if not already present in the subpopulation
					if(tableFitness.containsKey(newInd.getGenotype().toString())) {
						if(randgen.coin( 1 - ((generation*1.0) / maxOfGenerations) )) {
							if(!Utils.contains(bset.get(r), newInd)) {
								bset.get(r).add(newInd);
								exchanged++;
							}
						}
					}
					else {
						if(!Utils.contains(bset.get(r), newInd)) {
							bset.get(r).add(newInd);
							exchanged++;
						}
					}
				}
			}
			
			System.out.println(exchanged + " exchanged.");
			
			currInd = null;
			newInd = null;
			
			//Evaluate new individuals
			((MultipAbstractParallelEvaluator)evaluator).evaluateMultip(bset);
			
			//Select new individuals for subpopulations
			for(int i=0; i<numSubpop; i++) {
				if(bset.get(i).size() > subpopSize) {
					eSel = new EnsembleSelection(bset.get(i), subpopSize, nLabels, betaUpdatePop);
					eSel.setRandgen(randgen);
					eSel.selectEnsemble();
					bset.set(i, eSel.getEnsemble());
					eSel = null;
				}
			}
				
			break;
			
		case exchangeSubpop:
			/*
			 * For the better individuals in each subpop, try to copy to other subpopulations
			 * The probability of copy is biased by their position in the ensemble
			 */
			exchanged = 0;
			
			//Here store the exchanged inds (in order to not modify the bsets until the end)
			List<List<IIndividual>> exchangedInds = new ArrayList<List<IIndividual>>(numSubpop);
			for(int i=0; i<numSubpop; i++) {
				exchangedInds.add(i, new ArrayList<IIndividual>());
			}
			
			//For each subpop
			for(int p=0; p<numSubpop; p++) {
				//Each individual of the subpop
				for(int i=0; i<subpopSize; i++) {
					//Probability to be selected is lower as it is deeper in the ensemble
					//It is calculated as an exponential (importance ^ omega)
						//The greater the omega value, the greater the slope, so deeper individuals have less probability
					prob = Math.pow(((1.0*nClassifiers - i) / nClassifiers), omega);

					if(randgen.coin(prob)) {
						currInd = (MultipListIndividual) bset.get(p).get(i);
						
						//Select subpop different to the current one
						sp = currInd.getSubpop();
						do {
							r = randgen.choose(0, numSubpop);
						}while(r == sp);
						
						//Individual with same list of genotype, but different subpopulation
						newInd = new MultipListIndividual(new MultipListGenotype(r, new ArrayList<Integer>(currInd.getGenotype().genotype)));
						
						//If it has been evaluated yet at any moment
							//Probability to be included again is reduced with generations
						//If not, just include
						//Also, only include if it is already not in the corresponding subpop nor the exchanged inds
						if(tableFitness.containsKey(newInd.getGenotype().toString())) {
							if(randgen.coin( 1 - ((generation*1.0) / maxOfGenerations) )) {
								if(!Utils.contains(bset.get(r), newInd)) {
									if(!Utils.contains(exchangedInds.get(r), newInd)) {
										exchangedInds.get(r).add(newInd);
										exchanged++;
									}
								}
								
							}
						}
						else {
							if(!Utils.contains(bset.get(r), newInd)) {
								if(!Utils.contains(exchangedInds.get(r), newInd)) {
									exchangedInds.get(r).add(newInd);
									exchanged++;
								}
							}
						}
					}
				}
			}
			
			System.out.println(exchanged + " exchanged.");
			
			//Add exchanged inds to corresponding subpops
			for(int p=0; p<numSubpop; p++) {
				for(int i=0; i<exchangedInds.get(p).size(); i++) {
					bset.get(p).add(exchangedInds.get(p).get(i));
				}
			}
			
			currInd = null;
			newInd = null;
			exchangedInds = null;
			
			//Evaluate new individuals
			((MultipAbstractParallelEvaluator)evaluator).evaluateMultip(bset);
			
			//Select new individuals for subpopulations
			for(int i=0; i<numSubpop; i++) {
				if(bset.get(i).size() > subpopSize) {
					eSel = new EnsembleSelection(bset.get(i), subpopSize, nLabels, betaUpdatePop);
					eSel.setRandgen(randgen);
					eSel.selectEnsemble();
					bset.set(i, eSel.getEnsemble());
					eSel = null;
				}
			}
				
			break;
			
		case operators:
			/*
			 * Each individual is applied specific crossover or mutation given a probability
			 * These operator changes the subpopulations of the individuals
			 */
			
			//New created individuals
			ArrayList<MultipListIndividual> newInds = new ArrayList<MultipListIndividual>();
			IIndividual ind1, ind2;
			
			SubpopMutator subpopMut = new SubpopMutator();
			
			//Try for each individual
			for(int p=0; p<numSubpop; p++) {
				for(int i=0; i<subpopSize; i++) {
					//Try crossover
					if(randgen.coin(probCrossComm)) {
						ind1 = bset.get(p).get(i);
						
						//Select random ind for other subpop
						ind2 = Utils.randomIndDifferentSubpop(bset, ((MultipListIndividual)ind1).getSubpop(), randgen);
					
						//Cross ind and ind2
						MultipListIndividual [] crossed = ((Crossover)recombinator.getDecorated()).recombineInds((MultipListIndividual)ind1, (MultipListIndividual)ind2);
						newInds.add(crossed[0]);
						newInds.add(crossed[1]);
						
						crossed = null;
						ind1 = null;
						ind2 = null;
					}
					
					if(randgen.coin(probMutComm)) {
						ind1 = bset.get(p).get(i);
						
						//Mutate individual
						newInds.add(subpopMut.mutateInd((MultipListIndividual)ind1, numSubpop, randgen));
						
						ind1 = null;
					}
				}
			}
			
			//Add each new individual to its corresponding subpopulation (if applicable)
			int currSubpop;
			for(MultipListIndividual ind : newInds) {
				currSubpop = ind.getSubpop();
				
				//If the individual is not already included in the subpopulation
				if(!Utils.contains(bset.get(currSubpop), ind)) {
					//If individual was already evaluated at any moment, include with decreasing probability
					if(tableFitness.containsKey(ind.getGenotype().toString())) {
						if(randgen.coin( 1 - ((generation*1.0) / maxOfGenerations) )) {
							bset.get(currSubpop).add(ind);
						}
					}
					//If never evaluated, include in corresponding subpopulation
					else {
						bset.get(currSubpop).add(ind);
					}
				}
			}
			
			newInds = null;
			
			//Evaluate new individuals
			((MultipAbstractParallelEvaluator)evaluator).evaluateMultip(bset);
			
			//Select individuals in those subpops with more than allowed individuals
			for(int p=0; p<numSubpop; p++) {
				if(bset.get(p).size() > subpopSize) {
					//Update subpopulation with ensemble selection procedure
					eSel = new EnsembleSelection(bset.get(p), subpopSize, nLabels, betaUpdatePop);
					eSel.setRandgen(randgen);
					eSel.selectEnsemble();
					bset.set(p, eSel.getEnsemble());
					eSel = null;
				}
			}
			
		default:
			break;
		}
	
	}
	
	@Override
	protected void doControl()
	{
		System.out.println("Generation " + generation);
		
		if ((generation % generationsComm) == 0) {
			doCommunication();
		}
		
		if (generation >= maxOfGenerations) {
			if(prune) {
				int nPruned = bestEnsemble.prune(fullTrainData);
				System.out.println(nPruned + " members pruned.");
			}

			EnsembleEval eEval = new EnsembleEval(bestEnsemble, fullTrainData);
			double eFitness = eEval.evaluate();
			System.out.println("FINAL FITNESS: " + eFitness);
			
			state = FINISHED;
			return;
		}
	}	
}
