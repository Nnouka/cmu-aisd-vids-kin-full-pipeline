import { API_BASE_URL } from "../constants";

async function parseJson(response) {
  const payload = await response.json();
  return payload;
}

export async function fetchJob(jobId) {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
  const payload = await parseJson(response).catch(() => null);
  return {
    response,
    payload,
  };
}

export async function createJob(formData) {
  const response = await fetch(`${API_BASE_URL}/jobs`, {
    method: "POST",
    body: formData,
  });
  return {
    response,
    payload: await parseJson(response),
  };
}

export async function continueJob(jobId) {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/continue`, {
    method: "POST",
  });
  return {
    response,
    payload: await parseJson(response),
  };
}

export async function restartJob(jobId) {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/restart`, {
    method: "POST",
  });
  return {
    response,
    payload: await parseJson(response),
  };
}