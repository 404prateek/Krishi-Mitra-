const NDVIMapPage = () => {
  return (
    <div style={{ width: "100%", height: "calc(100dvh - 108px)", overflow: "hidden" }}>
      <iframe
        src="/ndvi-map.html"
        title="UP NDVI Satellite Map"
        width="100%"
        height="100%"
        style={{ border: "none", display: "block" }}
        allow="geolocation"
      />
    </div>
  );
};

export default NDVIMapPage;
