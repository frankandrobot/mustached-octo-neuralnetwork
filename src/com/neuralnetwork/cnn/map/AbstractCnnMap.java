package com.neuralnetwork.cnn.map;

import com.neuralnetwork.cnn.filter.IFilter;
import com.neuralnetwork.core.helpers.Dimension;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;

public abstract class AbstractCnnMap implements IMatrixNeuralLayer
{
    /**
     * the kernel is made of weights from this neuron.
     * the shared neuron contains most of the info necessary to build the layer
     */
    protected Neuron[] aSharedNeurons;
    protected DenseMatrix64F[] aKernels;

    /**
     * one dimension of the input. assumes input is square
     */
    protected int inputDim;

    /**
     * matrix used to store latest output
     */
    protected DenseMatrix64F output;

    /**
     * the filter used to convolve
     */
    protected IFilter[] filter;

    @Override
    public DenseMatrix64F generateY(DenseMatrix64F[] input)
    {
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        return generateOutput(new DenseMatrix64F[] { input });
    }

    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F[] input)
    {
        output = generateInducedLocalField(input);

        //apply activation function to output
        IActivationFunction phi = aSharedNeurons[0].phi();

        double[] _output = output.getData();

        for(int i=0; i<_output.length; i++)
        {
            _output[i] = phi.apply(_output[i]);
        }

        return output;
    }

    @Override
    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F[] input)
    {
        //first the weights
        for(int i=0; i<input.length; i++)
            filter[i].convolve(input[i], output);

        //now the biases
        double bias = aSharedNeurons[0].getBias();

        double[] o = output.getData();

        for(int i=0; i<o.length; i++)
        {
            o[i] += bias;
        }

        return output;
    }

    /**
     * @deprecated
     *
     * @return
     */
    public DenseMatrix64F getOutput()
    {
        return output;
    }

    @Override
    public int getNumberOfNeurons()
    {
        return 1;
    }

    @Override
    public DenseMatrix64F getWeightMatrix()
    {
        return null;
    }

    @Override
    public IActivationFunction.IDifferentiableFunction getImpulseFunction() {
        return null;
    }

    public Dimension getInputDim()
    {
        return new Dimension(inputDim, inputDim);
    }

    public Dimension getOutputDim()
    {
        return new Dimension(output.numRows, output.numCols);
    }
}
