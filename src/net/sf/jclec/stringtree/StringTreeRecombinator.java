package net.sf.jclec.stringtree;

import net.sf.jclec.ISpecies;

import net.sf.jclec.base.AbstractRecombinator;

/**
 * BinArrayIndividual (and subclasses) specific recombinator.  
 * 
 * @author Sebastian Ventura
 */

public abstract class StringTreeRecombinator extends AbstractRecombinator
{
	private static final long serialVersionUID = 7299906090922744418L;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------ Operation variables
	/////////////////////////////////////////////////////////////////

	/** Individual species (taked from execution context) */
	
	protected transient StringTreeSpecies species;

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty (default) constructor.
	 */
		
	public StringTreeRecombinator() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// AbstractRecombinator methods
	
	/** Set ppl=2 */
	
	@Override
	protected void setPpl() 
	{
		this.ppl = 2;
	}

	/** Set spl=2 */

	@Override
	protected void setSpl() 
	{
		this.spl = 2;
	}

	/**
	 * {@inheritDoc}
	 */
	
	@Override
	protected void prepareRecombination() 
	{
		ISpecies species = context.getSpecies();
		if (species instanceof StringTreeSpecies) {
			// Set individuals speciess
			this.species = (StringTreeSpecies) species;
		}
		else {
			throw new IllegalStateException("Invalid population species");
		}		
	}
	
	/* 
	 * Este mtodo fija el schema que vamos a utilizar para crear los genotipos
	 * de los nuevos individuos. Para ello, asegura que el objeto species que
	 * representa a los individuos de la poblacin es de tipo IBinArraySpecies.
	 * En caso negativo, lanza una excepcin.
	 */	
}
