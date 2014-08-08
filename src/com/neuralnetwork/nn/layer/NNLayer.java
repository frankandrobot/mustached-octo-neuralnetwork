package com.neuralnetwork.nn.layer;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class NNLayer implements INeuralLayer
{
    /**
     * for simplicity we assume all neurons have the same activation function
     */
    protected IActivationFunction.IDifferentiableFunction phi;

    /**
     * For simplicity we assume that all neurons have the same number of weights.
     * (the network is fully connected)
     * So:
     *
     * number of neurons x number of weights in a single neuron
     *
     */
    protected DenseMatrix64F weights;

    /**
     * number of weights in a single nueron x 1
     */
    protected DenseMatrix64F mInput;
    /**
     * number of neurons x 1
     */
    protected DenseMatrix64F mOutput;
    /**
     * (number of neurons + 1) x 1
     * with mY[0] = +1 always
     */
    protected double[] y;

    protected int numberOfNuerons;
    protected int numberOfWeightsInSingleNueron;

    NNLayer(NNLayerBuilder builder)
    {
        Neuron[] neurons = builder.aNeurons;

        numberOfNuerons = neurons.length;
        numberOfWeightsInSingleNueron = neurons[0].getNumberOfWeights();

        //get phi
        phi = neurons[0].phi();

        //build weight matrix = number of neurons x number of weights
        weights = new DenseMatrix64F(numberOfNuerons, numberOfWeightsInSingleNueron);

        for(int row = 0; row<weights.numRows; ++row)
            for(int col = 0; col<weights.numCols; ++col)
            {
                weights.unsafe_set(row,col, neurons[row].getWeight(col) );
            }

        //build input
        mInput = new DenseMatrix64F(numberOfWeightsInSingleNueron, 1);

        //build output
        mOutput = new DenseMatrix64F(numberOfNuerons, 1);

        //build y
        y = new double[numberOfNuerons + 1];
    }

    @Override
    public double[] generateY(double[] input)
    {
        double[] output = generateOutput(input);

        y[0] = 1;
        System.arraycopy(output,0, y,1, output.length);

        return y;
    }

    @Override
    public double[] generateOutput(double[] input)
    {

        double[] output = generateInducedLocalField(input);

        for(int i=0; i<output.length; ++i)
        {
            output[i] = phi.apply(output[i]);
        }

        return output;
    }

    @Override
    public double[] generateInducedLocalField(double[] input)
    {
        assert(input.length == getInputDim());
        assert(input[0] == 1.0);

        mInput.set(getInputDim(),1,false,input);
        CommonOps.mult(weights, mInput, mOutput);

        return mOutput.getData();
    }

    @Override
    public int getInputDim()
    {
        return numberOfWeightsInSingleNueron;
    }

    @Override
    public int getOutputDim()
    {
        return numberOfNuerons;
    }

    @Override
    public int getNumberOfNeurons()
    {
        return numberOfNuerons;
    }

    @Override
    public DenseMatrix64F getWeightMatrix()
    {
        return weights;
    }

    @Override
    public IActivationFunction.IDifferentiableFunction getImpulseFunction()
    {
        return phi;
    }
}
