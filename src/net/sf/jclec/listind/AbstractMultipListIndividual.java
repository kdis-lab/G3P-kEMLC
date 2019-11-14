package net.sf.jclec.listind;

import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;

import org.apache.commons.lang.builder.ToStringBuilder;

/**
 * MultipListIIndividual abstract implementation.
 *  
 * @author Jose M. Moyano
 * @author Sebastian Ventura
 */

public abstract class AbstractMultipListIndividual implements IIndividual 
{
	/////////////////////////////////////////////////////////////////
	// --------------------------------------------------- Attributes
	/////////////////////////////////////////////////////////////////


	/**
	 * 
	 */
	private static final long serialVersionUID = 8212975724348808961L;

	/** Individual genotype */
	
	protected MultipListGenotype genotype;
	
	/** Individual fitness */
	
	protected IFitness fitness;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty constructor 
	 */
	protected AbstractMultipListIndividual() 
	{
		super();
	}

	/**
	 * Constructor that sets individual genotype.
	 * 
	 * @param genotype Individual genotype
	 */
	protected AbstractMultipListIndividual(MultipListGenotype genotype) 
	{
		super();
		setGenotype(genotype);
	}

	/**
	 * Constructor that sets individual genotype and fitness. 
	 * 
	 * @param genotype Individual genotype
	 * @param fitness  Individual fitness
	 */
	protected AbstractMultipListIndividual(MultipListGenotype genotype, IFitness fitness) 
	{
		super();
		setGenotype(genotype);
		setFitness(fitness);
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////

	/**
	 * Sets individual genotype
	 * 
	 * @param genotype Individual genotype
	 */
	public final void setGenotype(MultipListGenotype genotype)
	{
		this.genotype = genotype;
	}

	/**
	 * Access to individual genotype
	 * 
	 * @return Individual genotype 
	 */
	public final MultipListGenotype getGenotype() 
	{
		return genotype;
	}

	// IIndividual interface

	/**
	 * {@inheritDoc}
	 */
	public final IFitness getFitness() 
	{
		return fitness;
	}

	/**
	 * {@inheritDoc}
	 */
	public final void setFitness(IFitness fitness) 
	{
		this.fitness = fitness;
	}

	
	/**
	 * Return an string that represent this fitness object legibly.
	 * 
	 * @return String that represents this individual.
	 */	
	@Override
	public String toString()
	{
		ToStringBuilder tsb = new ToStringBuilder(this);
		tsb.append("genotype", genotype.toString());
		tsb.append("fitness", fitness);
		return tsb.toString();
	}
}

