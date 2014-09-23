package com.neuralnetwork.cnn;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

/**
 * Don't confuse this with a {@link com.neuralnetwork.nn.layer.NNLayer}
 *
 * A single CNNLayer is composed of several {@link com.neuralnetwork.core.interfaces.IMatrixNeuralLayer}
 *
 */
class CNNLayer {

    List<MapInfo> lMaps = new LinkedList<MapInfo>();


    public HashSet<MapInfo> getInputMaps()
    {
        HashSet<MapInfo> hsInputMaps = new HashSet<MapInfo>();

        for(MapInfo mapInfo:lMaps)
            hsInputMaps.addAll(mapInfo.getInputMaps());

        return hsInputMaps;
    }
}
