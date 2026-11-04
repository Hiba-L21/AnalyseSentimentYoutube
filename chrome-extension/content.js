/**
 * Content Script — injecté dans les pages YouTube
 * Extrait les commentaires visibles dans le DOM
 */

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action !== "getComments") return;

  const selectors = [
    "#content-text",           // commentaire principal
    "yt-formatted-string#content-text",
  ];

  const nodes = document.querySelectorAll(selectors.join(", "));
  const comments = Array.from(nodes)
    .map(el => el.innerText.trim())
    .filter(text => text.length > 0);

  const unique = [...new Set(comments)];

  sendResponse({
    comments: unique,
    count: unique.length,
    url: window.location.href,
  });

  return true; // async response
});