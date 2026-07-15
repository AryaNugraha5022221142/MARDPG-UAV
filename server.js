const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('MARDPG-UAV server is running. You can run Python evaluation scripts in the terminal.');
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
