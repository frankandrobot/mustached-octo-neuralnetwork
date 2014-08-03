package com.neuralnetwork.core.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import org.ejml.data.DenseMatrix64F;

/**
 * On a single training example,
 * A single iteration of the backprop algorithm
 * on a single example
 */
class NNBackprop
{
    protected class OutputInfo
    {
        /**
         * each neuron has one
         */
        public double[] inducedLocalField;
        /**
         * each neuron has one
         */
        public double[] output;

        public OutputInfo(int numberOfNeuronsInLayer)
        {
            inducedLocalField = new double[numberOfNeuronsInLayer];
            output = new double[numberOfNeuronsInLayer];
        }
    }

    protected class GradientInfo
    {
        /**
         * each neuron has one
         */
        public double[] gradients;

        public GradientInfo(int numberOfNeuronsInLayer)
        {
            gradients = new double[numberOfNeuronsInLayer];
        }
    }

    protected Example example;

    protected INeuralLayer[] aLayers;

    protected OutputInfo[] aOutputInfo;
    protected GradientInfo[] aGradientInfo;

    protected DenseMatrix64F[] aCumulativeLearningTerms;

    NNBackprop(INeuralLayer... aLayers)
    {
        this.aLayers = aLayers;

        aOutputInfo = new OutputInfo[aLayers.length];
        aGradientInfo = new GradientInfo[aLayers.length];
        aCumulativeLearningTerms = new DenseMatrix64F[aLayers.length];

        for(int i=0; i<aLayers.length; i++)
        {
            int size = aLayers[i].getNumberOfNeurons();

            aOutputInfo[i] = new OutputInfo(size);
            aGradientInfo[i] = new GradientInfo(size);

            DenseMatrix64F matrix = aLayers[i].getWeightMatrix();

            aCumulativeLearningTerms[i] = new DenseMatrix64F(matrix.numRows, matrix.numCols);
        }
    }

    /**
     * On a single training example,
     * 1. Perform the forward propagation. Store induced local fields and impulse function values.
     * 2. Perform the backpropagation. Store each neuron's gradient.
     * 3. Calculate the learning term for each weight of each neuron
     * 4. Return a matrix containing the learning terms
     *
     * WARNING: this is NOT thread-safe!!!
     *
     * @param example
     * @return
     */
    DenseMatrix64F[] go(Example example)
    {
        this.example = example;

        forwardProp();
        backprop();

        return updateCumulativeLearningTerms();
    }

    protected void forwardProp()
    {
        setOutput(0, example.input);

        for(int i=1; i<aLayers.length; i++)
        {
            setOutput(i, aOutputInfo[i - 1].output);
        }
    }

    protected void setOutput(int index, double[] input)
    {
        double[] inducedLocalField = aLayers[index].generateInducedLocalField(input);

        aOutputInfo[index].inducedLocalField = inducedLocalField;

        IActivationFunction.IDifferentiableFunction phi = aLayers[index].getImpulseFunction();
        for(int i=0; i<inducedLocalField.length; i++)
            aOutputInfo[index].output[i] = phi.apply(inducedLocalField[i]);
    }

    /**
     * Requires that you call forwardProp first
     *
     * @return
     */
    protected void backprop()
    {
        for(int i=aLayers.length-1; i>=0 ;i--)
        {
            constructGradients(i);
        }
    }

    protected void constructGradients(int layerIndex)
    {
        INeuralLayer layer = aLayers[layerIndex];

        for(int neuronPos=0; neuronPos<layer.getNumberOfNeurons(); neuronPos++)
        {
            aGradientInfo[layerIndex].gradients[neuronPos] = gradient(layerIndex, neuronPos);
        }
    }

    /**
     * Gradient for the given example, layer, and neuron
     *
     * @param layer
     * @param neuron
     * @return gradient value for the given layer, and neuron
     */
    protected double gradient(int layer, int neuron)
    {
        OutputInfo outputInfo = aOutputInfo[layer];
        IActivationFunction.IDifferentiableFunction phi = aLayers[layer].getImpulseFunction();

        if (layer == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = outputInfo.inducedLocalField[neuron];
            final double output = outputInfo.output[neuron];

            return (example.expected[neuron] - output)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative(outputInfo.inducedLocalField[neuron])
                     * sumGradients(neuron, layer + 1);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron.
     *
     * The weights that connect neuron j are found in column j of the weight matrix.
     *
     * @param j neuron j
     * @param nextLayer
     *
     * @return
     */
    protected double sumGradients(int j, int nextLayer)
    {
        double rslt = 0f;
        DenseMatrix64F matrix = aLayers[nextLayer].getWeightMatrix();

        for(int neuron=0; neuron < matrix.numRows; ++neuron)
        {
            // w_ij^(l+1) * delta_i^(l+1)
            rslt += matrix.unsafe_get(neuron, j) * gradient(nextLayer, neuron);
        }

        return rslt;
    }

    protected DenseMatrix64F[] updateCumulativeLearningTerms()
    {
        //Δw^l_kj=ηδ^l_k * y^(l−1)_j
        for(int layer=0; layer<aLayers.length; ++layer) {

            DenseMatrix64F learningMatrix = aCumulativeLearningTerms[layer];

            for (int row = 0; row < learningMatrix.numRows; ++row)
                for (int col = 0; col < learningMatrix.numCols; ++col)
                {
                    double curTerm = learningMatrix.unsafe_get(row,col);

                    double gradient_k = aGradientInfo[layer].gradients[row];
                    double prevOutput_j = getPreviousOutput(layer-1, col);

                    learningMatrix.unsafe_set(row,col, curTerm + gradient_k * prevOutput_j);
                }
        }

        return aCumulativeLearningTerms;
    }

    protected double getPreviousOutput(int prevLayer, int neuron)
    {
        if (prevLayer == -1)
        {
            return example.input[neuron];
        }
        return aOutputInfo[prevLayer].output[neuron];
    }
}
