package com.neuralnetwork;

import com.neuralnetwork.core.IActivationFunctionTest;
import com.neuralnetwork.core.NVectorTest;
import com.neuralnetwork.core.neuron.MNeuronTest;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        IActivationFunctionTest.class,
        MNeuronTest.class,
        NVectorTest.class
})
public class WorkingTests
{

}
