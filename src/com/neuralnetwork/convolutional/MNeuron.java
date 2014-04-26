package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuron;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.Arrays;

/**
 * By convention, the bias is at the _end_ of the weights list
 *
 */
public class MNeuron implements INeuron<DenseMatrix64F>
{
    /**
     * A n x 1 matrix
     */
    DenseMatrix64F mWeights;
    double bias;
    IActivationFunction phi;

    DenseMatrix64F mDot;

    /**
     * The bias must be at the end
     *
     * @param phi activation function
     * @param aWeights 1D array of weights
     */
    public MNeuron(IActivationFunction phi, double... aWeights)
    {
        double[] withoutBias = Arrays.copyOf(aWeights, aWeights.length-1);
        this.mWeights = new DenseMatrix64F(aWeights.length-1,1,false,withoutBias);
        this.bias = aWeights[aWeights.length-1];
        this.mDot = new DenseMatrix64F(1,1);
        this.phi = phi;
    }

    /**
     * Induced local field
     *
     * @param input must be a 1xn matrix
     * @return
     */
    public double rawoutput(DenseMatrix64F input)
    {
        CommonOps.mult(input, mWeights, mDot);
        return mDot.unsafe_get(0,0) + bias;
    }

    /**
     * Value of activation function
     *
     * @param input must be a 1 x n matrix
     * @return
     */
    public double output(DenseMatrix64F input)
    {
        return phi.apply(rawoutput(input));
    }

    public double getWeight(int weight)
    {
        return weight < mWeights.numRows
                ? mWeights.unsafe_get(weight, 0)
                : bias;
    }

    public double getBias()
    {
        return bias;
    }

    public int getNumberOfWeights()
    {
        return mWeights.numRows + 1;
    }

    public IActivationFunction phi()
    {
        return phi;
    }

    public DenseMatrix64F getWeights()
    {
        return mWeights;
    }

    public void setWeight(int weight, double newWeight)
    {
        if (weight < mWeights.numRows)
            mWeights.unsafe_set(weight, 0, newWeight);
        else
            bias = newWeight;
    }

    @Override
    public String toString()
    {
        String rslt = String.format("%6.6g", mWeights.get(0));
        for(int i=1; i< mWeights.numRows; ++i)
        {
            rslt += "  "+String.format("%6.6g", mWeights.unsafe_get(i,0));
        }
        rslt += "  "+String.format("%6.6g", bias);
        return rslt;
    }
}
