package com.neuralnetwork.cnn.old;

import org.ejml.data.DenseMatrix64F;

public class OldConvolutionMapLayer extends OldFeatureMap
{

    public OldConvolutionMapLayer(Builder builder)
    {
        super(builder);

        outputClass.setMapInput(new DenseMatrix64F(1, receptiveFieldSize));
    }


    @Override
    protected DenseMatrix64F createFeatureMap(int inputSize)
    {
        final int n = inputSize - sqrtReceptiveFieldSize + 1;
        return new DenseMatrix64F(n,n);

    }

    @Override
    public double output(DenseMatrix64F input, int x, int y)
    {
        double inducedLocalField = rawoutput(input, x, y);
        return sharedNeuron.phi().apply(inducedLocalField);
    }

    @Override
    public double rawoutput(final DenseMatrix64F input, final int x, final int y)
    {
        outputClass.copy(input, sqrtReceptiveFieldSize, x, y);
        return sharedNeuron.rawoutput(outputClass.mapInput);
    }

    @Override
    public void output(DenseMatrix64F input, DenseMatrix64F aFeatureMap)
    {
        for(int i=0; i<=input.numRows - sqrtReceptiveFieldSize; i++)
            for(int j=0; j<=input.numCols - sqrtReceptiveFieldSize; j++)
            {
                //copy over input into data struct
                outputClass.copy(input, sqrtReceptiveFieldSize, i, j);
                //do it
                aFeatureMap.unsafe_set(i, j, sharedNeuron.output(outputClass.mapInput));
            }
    }

    @Override
    protected void disableWeightConnections(int[] aWeightConnections, int i, int j)
    {
        //find distance to borders
        final int distanceToL = j;
        final int distanceToT = i;
        final int distanceToR = sqrtReceptiveFieldSize - (oneDimInputSize - j);
        final int distanceToB = sqrtReceptiveFieldSize - (oneDimInputSize - i);

        //reset weights
        //for(int w=0; w<aWeightConnections.length; w++)
        //    aWeightConnections[w] = 1;

        //disable from left border
        for(int col=distanceToL+1; col<sqrtReceptiveFieldSize; col++)
            disableCol(aWeightConnections, col);
        //disable from right border
        for(int col=0; col<distanceToR; col++ )
            disableCol(aWeightConnections, col);
        //disable left border
        for(int row=distanceToT+1; row<sqrtReceptiveFieldSize; row++)
            disableRow(aWeightConnections, row);
        for(int row=0; row<distanceToB; row++)
            disableRow(aWeightConnections, row);
    }

    protected void disableCol(int[] aWeightConnections, int col)
    {
        for(int i=0; i<sqrtReceptiveFieldSize; i++)
            aWeightConnections[i*sqrtReceptiveFieldSize + col] = 0;
    }

    protected void disableRow(int[] aWeightConnections, int row)
    {
        for(int i=0; i<sqrtReceptiveFieldSize; i++)
            aWeightConnections[row*sqrtReceptiveFieldSize + i] = 0;
    }

    @Override
    public int featureMapColPosition(int weight, int j)
    {
        return j - weight % sqrtReceptiveFieldSize;
    }

    @Override
    public int featureMapRowPosition(int weight, int i)
    {
        return i - weight / sqrtReceptiveFieldSize;
    }

    @Override
    public DenseMatrix64F getOutput() {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }
}
