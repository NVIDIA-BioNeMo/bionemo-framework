import React, { memo } from 'react'
import FeatureCard from './FeatureCard'

const styles = {
  featureList: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    paddingRight: '8px',
    minHeight: 0,
  },
}

function FeatureListComponent({
  filteredFeatures,
  displayedCardCount,
  clickedFeatureId,
  features,
  cardResetKey,
  handleCardClick,
  loadExamples,
  vocabLogits,
  featureAnalysis,
  featureListRef,
  endOfListRef,
  featureRefs,
  onLoadMore,
}) {
  const visibleFeatures = filteredFeatures.slice(0, displayedCardCount)
  const clickedIsVisible = clickedFeatureId != null &&
    visibleFeatures.some(f => Number(f.feature_id) === Number(clickedFeatureId))
  const clickedFeature = clickedFeatureId != null && !clickedIsVisible
    ? features.find(f => Number(f.feature_id) === Number(clickedFeatureId))
    : null

  return (
    <div ref={featureListRef} style={styles.featureList}>
      {/* Only render clicked feature at top if NOT already in visible list */}
      {clickedFeature && (
        <FeatureCard
          key={`clicked-${clickedFeature.feature_id}-${cardResetKey}`}
          ref={el => { featureRefs.current[clickedFeature.feature_id] = el }}
          feature={clickedFeature}
          isHighlighted={true}
          forceExpanded={true}
          onClick={handleCardClick}
          loadExamples={loadExamples}
          vocabLogits={vocabLogits}
          featureAnalysis={featureAnalysis}
        />
      )}
      {visibleFeatures.map(feature => (
        <FeatureCard
          key={`${feature.feature_id}-${cardResetKey}`}
          ref={el => { featureRefs.current[feature.feature_id] = el }}
          feature={feature}
          isHighlighted={Number(clickedFeatureId) === Number(feature.feature_id)}
          forceExpanded={Number(clickedFeatureId) === Number(feature.feature_id)}
          onClick={handleCardClick}
          loadExamples={loadExamples}
          vocabLogits={vocabLogits}
          featureAnalysis={featureAnalysis}
        />
      ))}
      {/* Sentinel element for infinite scroll detection */}
      <div ref={endOfListRef} style={{ height: '1px' }} />
      {displayedCardCount < filteredFeatures.length && (
        <button
          onClick={onLoadMore}
          style={{
            padding: '10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px',
            cursor: 'pointer', background: 'transparent', border: '1px solid var(--border, #555)',
            borderRadius: '6px', width: '100%',
          }}
        >
          Load more — showing {visibleFeatures.length} of {filteredFeatures.length} (search/sort to find specific features)
        </button>
      )}
      {filteredFeatures.length === 0 && clickedFeatureId == null && (
        <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)' }}>
          No features match your selection.
        </div>
      )}
    </div>
  )
}

export default memo(FeatureListComponent)
