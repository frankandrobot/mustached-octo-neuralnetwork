package com.neuralnetwork.core.interfaces;

import org.ejml.data.DenseMatrix64F;

/**
 * Works in conjunction with {@link com.neuralnetwork.nn.backprop.NNBackprop}
 *
 * Assumes a specific relationship between input, output, neuron weights
 *
 * Example:
 * Input  Neuron Weights       Output  y
 * --------------------------------------
 *                                     1
 * I0     N1: w00 w01 w02 w03  O1      O1
 * I1     N2: w10 w11 w12 w13  O2      O2
 * I2     N3: w20 w21 w22 w23  O3      O3
 * I3
 *
 * 1. rows in weight matrix are a neuron's weights.
 *    columns in weight matrix match the input
 * 2. weights are in specific order:
 *    bias is first (column 1), maps to first input (always +1)
 *    the second weight (column 2) maps to the second input (first neuron output),
 *    the third weight (column 3) maps the third input (second neuron output),
 *    etc.
 * 3. the first value in the input is always 1
 *    (maps to the bias)
 * 4. output (including induced local field) is in specific order:
 *    first value is output of first neuron
 *    second value is output of second neuron,
 *    etc.
 * 5. y is in specific order:
 *    first value is always 1 (maps to bias),
 *    second value is output of first neuron,
 *    third value is output of second neuron,
 *    etc.
 * 6. input/outputs are 1D arrays but don't have to be.
 *
 * For simplicity we also assume that each neuron has the same activation function
 * (although it doesn't have to be).
 *
 * Again, we also arbitrarily define the first weight in a neuron to be the bias.
 */
public interface INeuralLayer extends IBasicLayer
{
    /**
     * y is the output with a +1 in the beginning for the bias
     *
     * we use the convention that the bias is at the beginning of the array i.e.,
     * input[0] = +1  AND
     * y[0] = +1
     *
     * @param input
     * @return
     */
    public double[] generateY(double[] input);

    /**
     * We use the convention that the bias is at the beginning of the array i.e.,
     * input[0] = +1
     *
     * @param input
     * @return
     */
    public double[] generateOutput(double[] input);

    /**
     * We use the convention that the bias is at the beginning of the array i.e.,
     * input[0] = +1
     *
     * @param input
     * @return
     */
    public double[] generateInducedLocalField(double[] input);


    public int getNumberOfNeurons();

    /**
     * Unfortunately, for this to work, the matrix needs to be in a specific format:
     * Neuron i's weights are in row i.
     *
     * Example:
     * N1: w00 w01 w02 w03
     * N2: w10 w11 w12 w13
     * N3: w20 w21 w22 w23
     *
     * @return
     */
    public DenseMatrix64F getWeightMatrix();

    /**
     * For simplicity, each neuron has the same impulse function.
     *
     * @return
     */
    public IActivationFunction.IDifferentiableFunction getImpulseFunction();
}
